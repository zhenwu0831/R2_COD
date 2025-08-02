import random
import os
import json
import dill
import numpy as np
from collections import defaultdict, Counter    
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer, AdamW, get_constant_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput
from dataloaders.ranker_dl import get_unified_dataloader
from models.text_model import T5EntityRanker
from models.graph_models import EnhancedGNNEncoder
from models.hybrid_model import HybridRankerModel
import argparse
import sys
import pdb
import wandb
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity
from torch.cuda.amp import autocast, GradScaler
import time
from utils.early_stopping import EarlyStopping
from utils.evaluation import *

# surpress warnings
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def kco_info_nce_loss(A, B, temp=0.1, allow_gradient=False):
    try:
        A = F.normalize(A, dim=1)
        B = F.normalize(B.detach(), dim=1) if not allow_gradient else F.normalize(B, dim=1)

        S = torch.matmul(A, B.T) 
        positives = torch.diag(S).unsqueeze(1) 
        mask = ~torch.eye(S.size(0), dtype=torch.bool).cuda()
        negatives = S[mask].view(S.size(0), -1) 
        logits = torch.cat([positives, negatives], dim=1) 
        labels = torch.zeros(S.size(0), dtype=torch.long).cuda()
        logits = logits / temp
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()

    return loss

def plot_pca(text_embs, node_embs, epoch, args):
    pca = PCA(n_components=2)
    all_embs = torch.cat([text_embs, node_embs], dim=0).cpu().detach().numpy()
    pca_result = pca.fit_transform(all_embs)
    
    text_pca = pca_result[:len(text_embs)]
    node_pca = pca_result[len(text_embs):]

    plt.figure(figsize=(8, 6))
    plt.scatter(text_pca[:, 0], text_pca[:, 1], label='Text Embs', alpha=0.6)
    plt.scatter(node_pca[:, 0], node_pca[:, 1], label='Graph Embs', alpha=0.6)
    plt.legend()
    plt.title(f'PCA of Text and Graph Embeddings at Epoch {epoch}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)

    os.makedirs(f"{args.output_dir}/imgs", exist_ok=True)

    img_path = f"{args.output_dir}/imgs/{args.wandb_run_name}_pca_epoch_{epoch}.png"
    plt.savefig(img_path)
    plt.close()

    wandb.log({f"PCA Plot at epoch {epoch}": wandb.Image(img_path)})

def compute_within_between_distances(embs1, embs2, metric="cosine"):
    """
    Computes within-group and between-group distances.
    
    Args:
        embs1 (torch.Tensor): First set of embeddings (text or graph)
        embs2 (torch.Tensor): Second set of embeddings (text or graph)
        metric (str): Distance metric ("cosine" or "euclidean")

    Returns:
        dict: { "within_group1": ..., "within_group2": ..., "between_groups": ... }
    """
    embs1 = embs1.cpu().detach().numpy()
    embs2 = embs2.cpu().detach().numpy()

    # Compute within-group distances
    within_group1 = cdist(embs1, embs1, metric).mean()
    within_group2 = cdist(embs2, embs2, metric).mean()

    # Compute between-group distances
    between_groups = cdist(embs1, embs2, metric).mean()

    return {
        "within_group1": within_group1,
        "within_group2": within_group2,
        "between_groups": between_groups,
    }

def calculate_cosine_similarity(projected1, projected2):
    """
    Calculate cosine similarity between two sets of embeddings.
    Handles varying batch sizes by matching the smaller batch size.
    """
    # Ensure embeddings are normalized for cosine similarity
    projected1 = F.normalize(projected1, p=2, dim=1)
    projected2 = F.normalize(projected2, p=2, dim=1)
    
    # Determine the minimum batch size
    min_size = min(projected1.size(0), projected2.size(0))
    
    # Truncate both tensors to match the smaller batch size
    projected1 = projected1[:min_size]
    projected2 = projected2[:min_size]
    
    # Compute cosine similarity (element-wise)
    cosine_sim = F.cosine_similarity(projected1, projected2, dim=1)
    return cosine_sim

def log_similarity_and_distances(text_rep, gnn_rep):
    # text_rep = F.normalize(text_rep, p=2, dim=1)
    # gnn_rep = F.normalize(gnn_rep, p=2, dim=1)

    distances = compute_within_between_distances(text_rep, gnn_rep)
    cosine_sim = calculate_cosine_similarity(text_rep, gnn_rep)

    wandb.log({
        "cosine_similarity": cosine_sim.mean().item(),
        "within_group_text": distances["within_group1"],
        "within_group_graph": distances["within_group2"],
        "between_groups": distances["between_groups"]
    })


def train(args, model, optimizer, train_loader, device, epoch, info_nce_temp=0.1, eps=1e-8):
    bce_loss = nn.BCELoss()
    model.train()
    running_loss = 0

    all_text_reps = []
    all_gnn_reps = []

    for step, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        logits, gnn_rep, text_rep, entity_labels, entity_mask = model(batch)

        all_text_reps.append(text_rep.mean(dim=1).detach().cpu())
        all_gnn_reps.append(gnn_rep.mean(dim=1).detach().cpu())

        loss_per_entity = bce_loss(logits, entity_labels.float().to(device))
        cl_loss = (loss_per_entity * entity_mask.float().to(device)).sum() / (entity_mask.sum())

        text_rep = text_rep.flatten(1, 2)
        gnn_rep = gnn_rep.flatten(1, 2)

        cod_loss = (kco_info_nce_loss(text_rep, gnn_rep, temp=info_nce_temp) +
                    kco_info_nce_loss(gnn_rep, text_rep, temp=info_nce_temp)) / 2

        text_rep = F.normalize(text_rep + eps, p=2, dim=1)
        gnn_rep = F.normalize(gnn_rep + eps, p=2, dim=1)

        wandb.log({"cl_loss": cl_loss.item(), "cod_loss": cod_loss.item(), "step": step})

        if args.use_mulco:
            loss = cl_loss + (cod_loss * 0.1)  # Adjust the weight of the COD loss as needed
            # loss = cod_loss + cl_loss
        else:
            # for params in model.module.gcn_head.parameters():
            #     params.requires_grad = False
            # for params in model.module.bert_head.parameters():
            #     params.requires_grad = False
            loss = cl_loss
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        del logits, gnn_rep, text_rep, entity_labels, entity_mask
        torch.cuda.empty_cache()
        wandb.log({"batch_loss": loss.item(), "step": step})
    avg_loss = running_loss / len(train_loader)

    all_text_reps = torch.cat(all_text_reps, dim=0)
    all_gnn_reps = torch.cat(all_gnn_reps, dim=0)
    log_similarity_and_distances(all_text_reps, all_gnn_reps)

    return avg_loss

def evaluate(args, model, eval_loader, device, split="dev"):
    model.eval()
    acc = 0
    num_examples = len(eval_loader.dataset)
    all_predictions = []
    all_gold_labels = []
    all_ids = []
    all_questions = []

    if split == "test":
        os.makedirs(args.output_dir, exist_ok=True)

    for batch in tqdm(eval_loader, desc="Evaluating"):
        with torch.no_grad():
            ID = batch["text_view"]["ID"]
            question = batch["text_view"]["question"]

            final_output, _, _, entity_labels, entity_mask = model(batch)

            final_output = final_output * entity_mask.float().to(device)  
            entity_labels = entity_labels * entity_mask.float().to(device)  

            all_predictions.extend(final_output.detach().cpu().tolist())
            all_gold_labels.extend(entity_labels.detach().cpu().tolist())
            all_ids.extend(ID)
            all_questions.extend(question)

            hits_k_score = compute_hits_at_k(final_output, entity_labels)
            acc += hits_k_score 

    if split == "test":
        # import pdb; pdb.set_trace()
        results = []
        for idx in range(len(all_ids)):
            results.append({
                "ID": all_ids[idx],
                "question": all_questions[idx],
                "preds": all_predictions[idx],
                "true": all_gold_labels[idx]
            })
        with open(f"{args.output_dir}/{split}_predictions.json", "w") as f:
            json.dump(results, f, indent=4)

    # wandb.log({f"{split}_hits@K": hits_k_score, "epoch": epoch})
    avg_acc = acc / num_examples
    return avg_acc  # Return the average accuracy (HITS@k score) for the evaluation set

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/webqsp", help="Directory containing the data")
    parser.add_argument("--freebase_dir", type=str, default="../data/freebase", help="Directory containing the freebase data")
    parser.add_argument("--wandb_project", type=str, default="Entity-Ranker")
    parser.add_argument("--wandb_run_name", type=str, default="run")

    parser.add_argument("--operation", type=str, default="train", help="Mode of operation: train, eval, or predict")
    # parser.add_argument("--setting",   type=str, default="multitask", help="Setting for combined loss: multitask or mulco")

    parser.add_argument("--dataset", type=str, default="webqsp", help="Dataset to use: webqsp or grailqa or cwq")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1e-5, help="Epsilon value for accuracy calculation")
    parser.add_argument("--info_nce_temp", type=float, default=0.1, help="Temperature for InfoNCE loss in combined mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./ckpts", help="Output directory for predictions")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")

    # Text model-specific arguments
    parser.add_argument("--generation_max_length", type=int, default=128, help="Maximum length for text generation")
    parser.add_argument("--input_max_length", type=int, default=1024, help="Maximum length for text input")
    parser.add_argument("--special_tokens", type=bool, default=True, help="Special tokens to add to the tokenizer")
    parser.add_argument("--text_emb_dim", type=int, default=768, help="Dimension of text embeddings")

    # Graph model-specific arguments
    parser.add_argument("--ptlm_model", type=str, default="t5-base", help="Pretrained transformer model for graph model")
    parser.add_argument("--gnn_depth", help="layers used in the gnn", type=int, default=2)
    parser.add_argument("--gnn_model", help="gnn_model", type=str, default='rgcn')
    parser.add_argument("--dropout", help="dropout used", type=float, default=0.2)
    parser.add_argument("--hidden_dim", help="hidden dimension", type=int, default=768)
    parser.add_argument("--max_seq_len", help="maximum sequence length", type=int, default=512)
    parser.add_argument("--node_emb_type", help="node embedding type", type=str, default='walklet')
    parser.add_argument("--patience", help="early stopping patience", type=int, default=5)

    parser.add_argument("--shared_space_dim", help="shared space dimension", type=int, default=768)
    parser.add_argument("--pca_interval", help="interval for PCA plot", type=int, default=3)

    parser.add_argument("--use_mulco", default=False, action='store_true', help="Use mulco loss instead of multitask")
    
    args = parser.parse_args()

    if 'base' in args.ptlm_model:
        args.max_seq_len = 512
        # args.node_emb_dim = 768 + 128 + 1
        args.node_emb_dim = 768

    return args

def main():
    args = get_args()
    device = seed_everything(args.seed)

    # Load graph data
    # graph_data_path = f'{args.data_dir}/KBQA_walklet_edge_label_768_dim_data.dill'
    # relations_dict_path = f'{args.data_dir}/relations_dict.json'
    graph_data_path = f'{args.data_dir}/KBQA_walklet_shuffled_768_dim_data.dill'
    relations_dict_path = f'{args.data_dir}/relations_dict_shuffled.json'

    with open(graph_data_path, 'rb') as f:
        graph_data = dill.load(f)
    with open(relations_dict_path, 'r') as f:
        relations_dict = json.load(f)

    args.num_rels = len(relations_dict)
    args.relations_path = relations_dict_path

    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    special_tokens = {
        'less': '<',  # Removed extra space to ensure proper tokenization
        'less_or_equal': '<='
    }

    # Adding entity tokens properly
    for i in range(98):
        special_tokens[f'E{i}'] = f'<E{i}>'  # Removed space before <E{i}> to ensure proper special token behavior

    # Registering them as special tokens
    special_tokens_list = list(special_tokens.values())
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_list})

    text_model = T5EntityRanker(tokenizer=tokenizer).to(device)
    graph_model = EnhancedGNNEncoder(args).to(device)
    model = HybridRankerModel(args, text_model, graph_model)
    model = torch.nn.DataParallel(model).to(device)  

    # Load datasets
    train_data = graph_data['train']
    dev_data = graph_data['dev']
    test_data = graph_data['test']

    unified_train_loader = get_unified_dataloader(train_data, args, batch_size=args.batch_size, shuffle=True, tokenizer=tokenizer)
    unified_dev_loader = get_unified_dataloader(dev_data, args, batch_size=args.batch_size, shuffle=False, tokenizer=tokenizer)
    unified_test_loader = get_unified_dataloader(test_data, args, batch_size=args.batch_size, shuffle=False, tokenizer=tokenizer)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = None

    # Initialize Weights & Biases
    wandb.login()
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    wandb.config.update(args)

    # Early stopping and best metrics tracking
    best_metrics = -float('inf')
    best_epochs = -1
    best_ckpt_path = f"{args.output_dir}/best_unified_model.pt"
    prediction_path = f"{args.output_dir}/unified_model_predictions.json"
    early_stopping = EarlyStopping(patience=args.patience)

    initial_ckpt_path = f"{args.output_dir}/initial_checkpoint.pt"
    # torch.save(model.state_dict(), initial_ckpt_path)

    model.eval()
    all_text_reps = []
    all_gnn_reps = []
    with torch.no_grad():
        for batch in tqdm(unified_train_loader, desc="Epoch 0 pretraining metrics"):
            _, gnn_rep, text_rep, _, _ = model(batch)
            # text_rep = text_rep.flatten(1, 2)
            # gnn_rep = gnn_rep.flatten(1, 2)
            text_rep = text_rep.mean(dim=1)
            gnn_rep = gnn_rep.mean(dim=1)
            all_text_reps.append(text_rep.cpu())
            all_gnn_reps.append(gnn_rep.cpu())
    all_text_reps = torch.cat(all_text_reps, dim=0)
    all_gnn_reps = torch.cat(all_gnn_reps, dim=0)
    log_similarity_and_distances(all_text_reps, all_gnn_reps)

    # Training
    if args.operation == "train":
        for epoch in range(args.epochs):
            train_loss = train(
                args, model, optimizer, unified_train_loader, device, epoch, info_nce_temp=args.info_nce_temp
            )
            wandb.log({"train_loss": train_loss, "epoch": epoch})

            print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}")

            eval_results = evaluate(
                args, model, unified_dev_loader, device, split="dev"
            )
            wandb.log({f"dev_hits@K": eval_results, "epoch": epoch})

            train_results = evaluate(
                args, model, unified_train_loader, device, split="train"
            )
            wandb.log({f"train_hits@K": train_results, "epoch": epoch})

            print(f"Epoch {epoch + 1}/{args.epochs}, Dev HITS@k: {eval_results:.4f}")

            if eval_results > best_metrics:
                best_metrics = eval_results
                best_epochs = epoch
                torch.save(model.state_dict(), best_ckpt_path)
            early_stopping(eval_results)
            if early_stopping.should_stop():
                print(f"Early stopping triggered for text model at epoch {epoch}")
                break

            if epoch != 0 and (epoch + 1 % 2 == 0):
                second_ckpt_path = f"{args.output_dir}/epoch{epoch}_checkpoint.pt"
                torch.save(model.state_dict(), second_ckpt_path)

    model.load_state_dict(torch.load(best_ckpt_path))
    test_results = evaluate(
        args, model, unified_test_loader, device, split="test"
    )
    wandb.log({f"test_hits@K": test_results})
    print(f"Final test results: {test_results}")

    wandb.finish()

if __name__ == "__main__":
    main()