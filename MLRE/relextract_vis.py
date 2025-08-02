from helper import *
from models import *
from dataloader import *
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import AdamW
import torch
import wandb
from visualization import *
import torch.nn.functional as F
# from igraph import *

def seed_everything():
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device


def get_model_name(model_name):
    

    model_dict = {
            'mbert-base'     : 'bert-base-multilingual-uncased',
            'xlmr-base'      : 'xlm-roberta-base',
            'infoxlm-base'   : 'microsoft/infoxlm-base',
            'rembert'        : 'google/rembert',
            'xlmr-large'     : 'xlm-roberta-large',
            'infoxlm-large'  : 'microsoft/infoxlm-large',
        }

    return model_dict[model_name]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang",       help="choice of source language",   type=str, default='en')
    parser.add_argument("--tgt_lang",       help="choice of target language",   type=str, default='en')

    parser.add_argument("--mode",      		help="choice of operation",       	type=str, default='eval')

    
    parser.add_argument("--model_name",     help="choice of multi-lingual model",type=str, default='mbert-base')
    parser.add_argument("--dep_model",      help="choice of dependency model",   type=str, default='stanza')

    parser.add_argument("--seed", 			help="random seed", 				type=int, default=98105)
    parser.add_argument("--gpu", 			help="choice of device", 			type=str, default='0')
    
    parser.add_argument("--gnn_depth", 		help="layers used in the gnn", 		type=int, default =2)
    parser.add_argument("--gnn_model",      help="gnn_model", 					type=str, default='rgcn')
    parser.add_argument("--drop", 			help="dropout_used", 				type=float, default=0.2)

    ## dependent on model_configuration
    parser.add_argument("--node_emb_dim", 	help="number of unseen classes", 	type=int, default=768)
    parser.add_argument("--head_emb_dim", 	help="number of unseen classes", 	type=int, default=2048)
    parser.add_argument("--max_seq_len",    help="maximum sequence length",  	type=int, default=512)

    parser.add_argument("--dep", 			help="dependency_parsing", 			type=str, default='1')
    
    
    parser.add_argument('--dataset', 	    help='choice of dataset', 			type=str, default='redfm')
    parser.add_argument('--connection', 	help='connection', 					type=str, default='mulco_combined')	
    parser.add_argument('--setting', 	    help='choice of graph/text/both', 	type=str, default='both')	
    parser.add_argument("--batch_size", 										type=int, default=4)

    # default parameters
    parser.add_argument("--epochs", 											type=int, default=60)
    parser.add_argument("--patience", 											type=int, default=10)
    parser.add_argument("--lr", 		    help="learning rate", 				type=float, default=1e-5)
    parser.add_argument('--temperature',    help='temperature for mulco', 		type=float, default=0.1)

    ## parameters for infoxlm
    # parser.add_argument("--epochs", 											type=int, default=60)
    # parser.add_argument("--patience", 											type=int, default=15)
    # parser.add_argument("--lr", 		    help="learning rate", 				type=float, default=2e-5)


    args                  = parser.parse_args()
    
    args.ml_model         = get_model_name(args.model_name)

    if 'base'  in args.ml_model:
        args.node_emb_dim = 768
        args.max_seq_len  = 512

    if 'rembert' in args.ml_model:
        args.node_emb_dim = 1152
        args.max_seq_len  = 512
    
    return args


def seen_eval(model, loader, device):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [],[]
 
    for data in tqdm(loader):
        try:
            bat = {}
            for k, v in data.items():
                bat[k] = v.to(device)


            with torch.no_grad():
                results				= model(bat)
                rel_logits          = results['rel_logits']
            
            _, pred 				= 	torch.max(rel_logits, 1)
            _, labels 				= 	torch.max(bat['label_ids'], 1)

            y_pred.extend(list(np.array(pred.cpu().detach())))
            y_true.extend(list(np.array(labels.cpu().detach())))
        
        except Exception as e:
            continue
        
            
    f1		  					= 	f1_score(y_true,y_pred, average="macro")
    p1, r1 						= 	precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro')
 
    return p1, r1, f1
 
def store_visualizations(model, loader, device, args, epoch, split='train'):
    eps = 1e-10

    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [],[]

    text_reps, node_reps = [], []
 
    for data in tqdm(loader):
        bat = {}
        for k, v in data.items():
            bat[k] = v.to(device)


        with torch.no_grad():
            results				= model(bat)
            # 'text_ent_embs': text_ent_embs,
            # 'graph_ent_embs': graph_ent_embs,
            text_embs           = results['text_ent_embs']
            node_embs           = results['graph_ent_embs']
            text_reps.append(text_embs.cpu().detach())
            node_reps.append(node_embs.cpu().detach())

            rel_logits          = results['rel_logits']

        _, pred 				= 	torch.max(rel_logits, 1)
        _, labels 				= 	torch.max(bat['label_ids'], 1)
        y_pred.extend(list(np.array(pred.cpu().detach())))
        y_true.extend(list(np.array(labels.cpu().detach())))
    

    text_reps           = torch.cat(text_reps, dim=0)  
    node_reps           = torch.cat(node_reps, dim=0)  

    text_reps           = text_reps + 1e-10
    node_reps           = node_reps + 1e-10

    ## Save the PCA PLOTS for the unnormalized text/node representations
    img_path                = f"{args.img_dir}/{split}_pca_epoch_{epoch}.png"
    args.img_path           = img_path
    plot_pca(text_reps, node_reps, epoch, args)

    ## Save the PCA PLOTS for the normalized text/node representations
    norm_text_reps         = F.normalize(text_reps, p=2, dim=1)
    norm_node_reps         = F.normalize(node_reps, p=2, dim=1)
    img_path               = f"{args.img_dir}/{split}_pca_epoch_{epoch}_normalized.png"
    args.img_path          = img_path
    plot_pca(norm_text_reps, norm_node_reps, epoch, args)


    results_dict_fp   = f'{args.img_dir}/{split}_results_dict_epoch_{epoch}.json'    
    cosine_sim          = calculate_cosine_similarity(text_reps, node_reps).mean().item()
    dist_dict           = compute_within_between_distances(text_reps, node_reps, metric="cosine")
    dist_dict['cosine_sim'] = cosine_sim

    with open(results_dict_fp, 'w') as f:
        json.dump(dist_dict, f, indent=4)

    return dist_dict
 

def seen_predict(model, loader, device):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [],[]
 
    for data in tqdm(loader):
        try:
            bat = {}
            for k, v in data.items():
                bat[k] = v.to(device)


            with torch.no_grad():
                results				= model(bat)
                rel_logits          = results['rel_logits']
            
            _, pred 				= 	torch.max(rel_logits, 1)
            _, labels 				= 	torch.max(bat['label_ids'], 1)

            y_pred.extend(list(np.array(pred.cpu().detach())))
            y_true.extend(list(np.array(labels.cpu().detach())))
        
        except Exception as e:
            continue
        
            
    f1		  					= 	f1_score(y_true,y_pred, average="macro")
    p1, r1 						= 	precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro')
 
    return {'y_true': y_true, 'y_pred': y_pred, 'f1': f1, 'precision': p1, 'recall': r1}
 

def add_optimizer(model, train_len ):
    warmup_proportion 	= 0.05
    n_train_steps		= int(train_len/args.batch_size) * args.epochs
    num_warmup_steps	= int(float(warmup_proportion) * float(n_train_steps))
    param_optimizer		= list(model.named_parameters())
    param_optimizer		= [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay			= ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr= args.lr)

    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=n_train_steps)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
    return optimizer, scheduler


def main(args):
    device 												=   seed_everything()
    args.data_dir										=   f'../data/{args.dataset}'

    ############## Load the model and data ######################
    
    for lang in ['en', 'es', 'fr', 'de', 'it']:
        args.src_lang 									    = 	lang
        args.tgt_lang 									    = 	lang
    
        src_file 											= 	f'{args.data_dir}/{args.src_lang}_{args.model_name}_{args.dep_model}_combined.dill'
        tgt_file 											= 	f'{args.data_dir}/{args.tgt_lang}_{args.model_name}_{args.dep_model}_combined.dill'

        # src_file 											= 	f'../../multilingual-re/data/{args.dataset}/{args.src_lang}_{args.model_name}_{args.dep_model}.dill'
        # tgt_file 											= 	f'../../multilingual-re/data/{args.dataset}/{args.tgt_lang}_{args.model_name}_{args.dep_model}.dill'


        # print(args)

        deprel_dict 										= 	load_deprels(enhanced=False)

        relation_dict                                       =   load_json(f'{args.data_dir}/relation_dict.json')

        args.num_rels 										= 	len(relation_dict)
        
        args.num_deps 										= 	len(deprel_dict)    

        print('Number of relations: ', args.num_rels)
        print('Number of dependencies: ', args.num_deps)
        
        print(args)
        
        if check_file(src_file):
            src_dataset										=   load_dill(src_file)
        else:
            print('SRC FILE IS NOT CREATED'); exit()
    
        if check_file(tgt_file):
            tgt_dataset										=   load_dill(tgt_file)
        else:
            print('TGT FILE IS NOT CREATED'); exit()

    
        if args.src_lang == args.tgt_lang:
            train_data, dev_data, test_data     			=   src_dataset['train'], src_dataset['validation'], src_dataset['test']
        else:
            train_data, dev_data, test_data     			=   src_dataset['train'], tgt_dataset['validation'], tgt_dataset['test']

        print('train size: {}, dev size {}, test size: {}'.format(len(train_data), len(dev_data), len(test_data)))
        print('Data is successfully loaded')
        

        if args.connection	    == 'residual':
            model 											= MLRelResidualClassifier(args)
        elif args.connection	== 'concat':
            model											= MLRelConcatClassifier(args)
        elif args.connection	== 'mulco':
            model                                           = MLRelMulcoClassifier(args)
        elif args.connection    == 'mulco_doc':
            model                                           = MLRelMulcoDocClassifier(args)    
        elif args.connection   == 'mulco_combined':            
            model                                           = MLRelMulcoCombinedClassifier(args)

        
        model       										= model.to(device)

        ce_loss 											= nn.CrossEntropyLoss()

        trainset    										= RelDataset(train_data, args)

        ##### loading the dataset for training and evaluation ########

        trainloader 										= DataLoader(trainset, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle=True)
        model.train()
    
        optimizer 											= torch.optim.AdamW(model.parameters(), lr=args.lr)
        # optimizer, scheduler 								= add_optimizer(model, len(train_data))
    
        # checkpoint_file 								    =  f'../ckpts/{args.dataset}/{args.src_lang}_{args.src_lang}-model_{args.model_name}-parser_{args.dep_model}-gnn_{args.gnn_model}-connection_{args.connection}-setting_{args.setting}-gnn-depth_{args.gnn_depth}-seed_{args.seed}'

        file_format                                         = f'{args.dataset}/{args.src_lang}_{args.tgt_lang}-model_{args.model_name}-parser_{args.dep_model}-gnn_{args.gnn_model}-connection_{args.connection}-setting_{args.setting}-gnn-depth_{args.gnn_depth}-head_dim_{args.head_emb_dim}-seed_{args.seed}'

        
        checkpoint_file 								    =  f'../ckpts/{file_format}'


        best_f1, best_model 								= 0, None
    
        
        
        if args.setting == 'both' and 'mulco' in args.connection:
            img_dir                                         =  f'../epochwise_images/{file_format}'
            args.img_dir 								    = img_dir
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)


        devset 											= RelDataset(dev_data, args)
        devloader     							   		= DataLoader(devset, batch_size=args.batch_size, collate_fn=create_mini_batch)
        kill_cnt 										= 0

        if args.setting == 'both' and 'mulco' in args.connection:
            
            dist_dict = store_visualizations(model, devloader, device, args, epoch='None', split='dev')
            dist_dict = store_visualizations(model, trainloader, device, args, epoch='None', split='train')
            
            
    

if __name__ =='__main__':	
    args                            =   get_args()
    main(args)
