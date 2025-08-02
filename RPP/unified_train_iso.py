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
from transformers import AutoTokenizer, AdamW, get_constant_schedule_with_warmup
from unified_dataloader import get_unified_dataloader, UnifiedDataset, compute_global_label_map
from models.graph_models import GNNClassifier
from baselines.text import BERTBaseline, T5Baseline
from models.hybrid_model import HybridModel  
import argparse
import sys
import pdb
import wandb
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from utils.shared_projection import SharedProjection
from torch.cuda.amp import autocast, GradScaler
import time
from utils.early_stopping import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
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

def add_optimizer(model, train_len, args):
    warmup_proportion = 0.05
    n_train_steps = int(train_len / args.batch_size) * args.epochs
    num_warmup_steps = int(float(warmup_proportion) * float(n_train_steps))
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-4},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
    
    return optimizer, scheduler

def add_optimizers(text_model, graph_model, train_len, args):
    warmup_proportion = 0.05
    n_train_steps = int(train_len / args.batch_size) * args.epochs
    num_warmup_steps = int(float(warmup_proportion) * float(n_train_steps))
    graph_param_optimizer = list(graph_model.named_parameters())
    text_param_optimizer = list(text_model.named_parameters())
    # combine the parameters
    param_optimizer = graph_param_optimizer + text_param_optimizer

    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-4},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
    
    return optimizer, scheduler

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

    os.makedirs(f"{args.output_path}/imgs", exist_ok=True)

    img_path = f"{args.output_path}/imgs/{args.run_name}_pca_epoch_{epoch}.png"
    plt.savefig(img_path)
    plt.close()

    wandb.log({f"PCA Plot at epoch {epoch}, batch 0": wandb.Image(img_path)})

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


def train(args, model, optimizer, train_loader, device, epoch):
    model.train()
    running_loss = 0
    all_preds, all_labels = [], []
    all_z_gcn_reps = []
    all_z_bert_reps = []

    for step, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        try:
            logits, z_gcn_reps, z_bert_reps = model(batch)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
        labels = batch['text_view']['iso_labels'].to(device)

        cl_loss = F.cross_entropy(logits, labels)
        cod_loss = (kco_info_nce_loss(z_gcn_reps, z_bert_reps) + kco_info_nce_loss(z_bert_reps, z_gcn_reps)) / 2

        wandb.log({"cl_loss": cl_loss.item(), "cod_loss": cod_loss.item()})

        if step == 0:  # Only plot PCA for the first batch of each epoch
            plot_pca(z_bert_reps, z_gcn_reps, epoch, args)

        if args.use_mulco:
            loss = cl_loss + cod_loss
        else:
            loss = cl_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = labels.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

        all_z_gcn_reps.append(z_gcn_reps.cpu())
        all_z_bert_reps.append(z_bert_reps.cpu())

        wandb.log({"train_loss": loss.item()})

    # Concatenate all representations
    all_z_gcn_reps = torch.cat(all_z_gcn_reps, dim=0)
    all_z_bert_reps = torch.cat(all_z_bert_reps, dim=0)

    distances = compute_within_between_distances(all_z_bert_reps, all_z_gcn_reps)
    wandb.log({
        "within_group_text": distances["within_group1"],
        "within_group_graph": distances["within_group2"],
        "between_groups": distances["between_groups"],
    })

    cosine_sim = calculate_cosine_similarity(all_z_bert_reps, all_z_gcn_reps)
    wandb.log({"cosine_similarity": cosine_sim.mean().item()})

    avg_loss = running_loss / len(train_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    wandb.log({"batch_loss": avg_loss, "train_f1": f1})

    return avg_loss, f1


def evaluate(model, eval_loader, device):
    model.eval()
    all_preds, all_labels, all_questions, all_ids = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits, _, _ = model(batch)
            labels = batch['text_view']['iso_labels'].to(device)
            question = batch['text_view']['question']
            ids = batch['text_view']['ID']

            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_questions.extend(question)
            all_ids.extend(ids)

    f1 = f1_score(all_labels, all_preds, average='macro')
    return f1, all_preds, all_labels, all_questions, all_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--rel_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--input_max_length", type=int, default=512)
    parser.add_argument("--max_seq_len", help="maximum sequence length", type=int, default=512)
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--ptlm_model", type=str, default="bert-base-uncased")
    parser.add_argument("--gnn_model", type=str, default="rgcn")
    parser.add_argument("--gnn_depth", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--prediction_dir_path", type=str, default=None)
    parser.add_argument("--use_mulco", action='store_true', help="Use mulco for training")
    parser.add_argument("--run_name", type=str, default="unified_model")
    parser.add_argument("--pca_interval", help="interval for PCA plot", type=int, default=3)

    args = parser.parse_args()

    wandb.login()
    wandb.init(project="TG_ISO_new", entity="ANON", name=args.run_name)

    device = seed_everything(args.seed)

    # Load dataset
    with open(args.data_path, 'rb') as f:
        dataset = dill.load(f)
    
    with open(args.rel_path, 'r') as f:
        relations_dict = json.load(f)

    if "gpt2" in args.ptlm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.ptlm_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.ptlm_model)
    
    train_dataset, dev_dataset, test_dataset = dataset['train'], dataset['dev'], dataset['test']

    global_label_map, num_classes = compute_global_label_map(train_dataset, dev_dataset, test_dataset)
    print(global_label_map)
    args.num_classes = num_classes
    args.num_rels = len(relations_dict)

    train_loader = get_unified_dataloader(train_dataset, args, batch_size=args.batch_size, shuffle=True, global_label_map=global_label_map, tokenizer=tokenizer)
    eval_loader = get_unified_dataloader(dev_dataset, args, batch_size=args.batch_size, shuffle=False, global_label_map=global_label_map, tokenizer=tokenizer)
    test_loader = get_unified_dataloader(test_dataset, args, batch_size=args.batch_size, shuffle=False, global_label_map=global_label_map, tokenizer=tokenizer)

    graph_model = GNNClassifier(args, tokenizer).to(device)
    text_model = BERTBaseline(args, tokenizer=tokenizer).to(device) if "bert" in args.ptlm_model else T5Baseline(args, tokenizer=tokenizer).to(device)
    model = HybridModel(args, text_model, graph_model).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    best_f1 = 0
    best_ckpt_path = f"{args.output_path}/best_unified_model.pt"
    prediction_path = f"{args.output_path}/unified_model_predictions.json"

    early_stopping = EarlyStopping(patience=args.patience)

    def compute_epoch_embedding_stats(model, loader, device, args, epoch_label="init"):
        model.eval()
        all_z_gcn_reps = []
        all_z_bert_reps = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Logging {epoch_label} stats"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                _, z_gcn_reps, z_bert_reps = model(batch)
                all_z_gcn_reps.append(z_gcn_reps.detach().cpu())
                all_z_bert_reps.append(z_bert_reps.detach().cpu())

        all_z_gcn_reps = torch.cat(all_z_gcn_reps, dim=0)
        all_z_bert_reps = torch.cat(all_z_bert_reps, dim=0)

        if args.pca_interval == 0:
            plot_pca(all_z_bert_reps, all_z_gcn_reps, epoch_label, args)

        distances = compute_within_between_distances(all_z_bert_reps, all_z_gcn_reps)
        wandb.log({
            "within_group_text": distances["within_group1"],
            "within_group_graph": distances["within_group2"],
            "between_groups": distances["between_groups"],
        })
        cosine_sim = calculate_cosine_similarity(all_z_bert_reps, all_z_gcn_reps)
        wandb.log({"cosine_similarity": cosine_sim.mean().item()})

    compute_epoch_embedding_stats(model, train_loader, device, args, epoch_label="epoch_0_pretrain")

    
    if args.mode == "train":
        for epoch in range(1,args.epochs+1):
            wandb.log({"epoch": epoch})
            train_loss, train_f1 = train(args, model, optimizer, train_loader, device, epoch)
            eval_f1, _, _, _, _ = evaluate(model, eval_loader, device)

            wandb.log({"eval_f1": eval_f1})

            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Eval F1: {eval_f1:.4f}")

            if eval_f1 > best_f1:
                best_f1 = eval_f1
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"🔹 New Best Model Found! Saving to {best_ckpt_path}")
                print(f"Saved Best Model with Eval F1: {best_f1:.4f}")

            early_stopping(eval_f1)
            if early_stopping.should_stop():
                print(f"Early stopping triggered for text model at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_ckpt_path))
    test_f1, all_preds, all_labels, all_questions, all_ids = evaluate(model, test_loader, device)
    print(f"Final Test F1: {test_f1:.4f}")
    wandb.log({"test_f1": test_f1})

    predictions = []
    if args.prediction_dir_path is not None:
        prediction_path = f"{args.prediction_dir_path}/unified_model_predictions.json"
    with open(prediction_path, 'w') as f:
        for i, question, pred, label in zip(all_ids, all_questions, all_preds, all_labels):
            # f.write(f"Question: {question}, Predicted: {pred}, True: {label}\n")
            predictions.append({"id": i, "question": question, "predicted": int(pred), "true": int(label)})
        json.dump(predictions, f, indent=4)

    wandb.finish()

if __name__ == "__main__":
    main()
