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

    parser.add_argument("--seed", 			help="random seed", 				type=int, default=11737)
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
 

# def seen_predict(model, loader, device):
#     model.eval()
#     correct, total = 0, 0
#     y_true, y_pred = [],[]
 
#     for data in tqdm(loader):
#         bat = {}
#         for k, v in data.items():
#             bat[k] = v.to(device)
            
#         with torch.no_grad():
#             results									= model(bat)
#             rel_logits                          	= results['rel_logits']
        
#         _, pred 				= 	torch.max(rel_logits, 1)
#         _, labels 				= 	torch.max(bat['label_ids'], 1)
  
#         y_pred.extend(list(np.array(pred.cpu().detach())))
#         y_true.extend(list(np.array(labels.cpu().detach())))

#     return y_pred, y_true


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
    
    src_file 											= 	f'{args.data_dir}/{args.src_lang}_{args.model_name}_{args.dep_model}_combined.dill'
    tgt_file 											= 	f'{args.data_dir}/{args.tgt_lang}_{args.model_name}_{args.dep_model}_combined.dill'

    # src_file 											= 	f'../../multilingual-re/data/{args.dataset}/{args.src_lang}_{args.model_name}_{args.dep_model}.dill'
    # tgt_file 											= 	f'../../multilingual-re/data/{args.dataset}/{args.tgt_lang}_{args.model_name}_{args.dep_model}.dill'

    # import pdb; pdb.set_trace()

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
 
    if args.mode 										== 'train':
        wandb.login()
        wandb.init(project="multilingual_re", entity="flow-graphs-ORG", name=f'{checkpoint_file.split("/")[-1]}')
    
        wandb.config.update(args)

        if args.setting == 'both':
            img_dir                                         =  f'../epochwise_images/{file_format}'
            args.img_dir 								    = img_dir
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)


        devset 											= RelDataset(dev_data, args)
        devloader     							   		= DataLoader(devset, batch_size=args.batch_size, collate_fn=create_mini_batch)
        kill_cnt 										= 0

        if args.setting == 'both' :
            try:
                dist_dict = store_visualizations(model, devloader, device, args, epoch='None', split='dev')
                wandb.log({"dev_within_group1": dist_dict['within_group1']})
                wandb.log({"dev_within_group2": dist_dict['within_group2']})
                wandb.log({"dev_between_groups": dist_dict['between_groups']})
                wandb.log({"dev_cosine_sim": dist_dict['cosine_sim']})

                dist_dict = store_visualizations(model, trainloader, device, args, epoch='None', split='train')
                wandb.log({"train_within_group1": dist_dict['within_group1']})
                wandb.log({"train_within_group2": dist_dict['within_group2']})
                wandb.log({"train_between_groups": dist_dict['between_groups']})
                wandb.log({"train_cosine_sim": dist_dict['cosine_sim']})

            except Exception as e:
                print('Error in storing visualizations')
                import pdb; pdb.set_trace()
                print(e)
                
        

        for epoch in range(args.epochs):
            print(f'============== TRAIN ON THE {epoch+1}-th EPOCH ==============')
            running_loss, correct, total = 0.0, 0, 0
            
            text_reps, node_reps                        = [], []

            model.train()

            batch_cnt                                       = 0
            for data in tqdm(trainloader):
                batch_cnt                                   += 1
                try:
                    bat = {}
                    for k, v in data.items():
                        bat[k] = v.to(device)
                except Exception as e:
                    print(e)
                    # import pdb; pdb.set_trace()

                optimizer.zero_grad()

                results 							    = model(bat)

                rel_logits                           	= results['rel_logits']

                loss 								    = (ce_loss(rel_logits.view(-1, args.num_rels), bat['label_ids'].float()))

                if args.setting == 'both':
                    cl_loss                             = results['mulco_loss']
                    if cl_loss is not None:
                        loss                            += results['mulco_loss']    

                wandb.log({"batch_loss": loss.item()})
                loss.backward()
                optimizer.step()
                running_loss += loss.item()


            print('============== EVALUATION ON DEV DATA ==============')

            wandb.log({"loss": running_loss})
            wandb.log({"data": args.dataset})
            
            # pt, rt, f1t 	 = seen_eval(model, trainloader, device=device)
            # print(f'Eval data {f1t} \t Prec {pt} \t Rec {rt}')

            pt, rt, f1t 	 = seen_eval(model, devloader, device=device)

            wandb.log({"dev_f1": f1t})
            print(f'Eval data \t Prec {pt} \t Rec {rt} \t F1 {f1t}')

            if f1t >= best_f1:
                best_p, best_r, best_f1 = pt, rt, f1t
                wandb.log({"best_f1": best_f1})
                kill_cnt    = 0
                best_f1     = f1t
                torch.save(model.state_dict(),checkpoint_file)
            else:
                kill_cnt +=1
                if kill_cnt >= args.patience:
                    break
            
            print(f'[best val] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}')

            if args.setting == 'both':
                try:
                    dist_dict = store_visualizations(model, devloader, device, args, epoch, split='dev')
                    wandb.log({"dev_within_group1": dist_dict['within_group1']})
                    wandb.log({"dev_within_group2": dist_dict['within_group2']})
                    wandb.log({"dev_between_groups": dist_dict['between_groups']})
                    wandb.log({"dev_cosine_sim": dist_dict['cosine_sim']})

                    dist_dict = store_visualizations(model, trainloader, device, args, epoch, split='train')
                    wandb.log({"train_within_group1": dist_dict['within_group1']})
                    wandb.log({"train_within_group2": dist_dict['within_group2']})
                    wandb.log({"train_between_groups": dist_dict['between_groups']})
                    wandb.log({"train_cosine_sim": dist_dict['cosine_sim']})

                except Exception as e:
                    print('Error in storing visualizations')
                    import pdb; pdb.set_trace()
                    print(e)
                    continue
                    
                

        testset 												= RelDataset(test_data, args)
        testloader     							   				= DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)
        model.load_state_dict(torch.load(checkpoint_file))
        model.eval()
        pt, rt, test_f1	 										= seen_eval(model, testloader, device=device)
        wandb.log({"test_f1": test_f1})


    # Evaluation is done here. 


    if args.mode											== 'eval':

        checkpoint_file 								    =  f'../ckpts/{file_format}'


        if not check_file(checkpoint_file) :
            print('MODEL CANNOT BE LOADED BY THESE SPECIFICATIONS')
            return

        wandb.login()
        wandb.init(project="multilingual_re", entity="flow-graphs-ORG", name=f'{checkpoint_file.split("/")[-1]}')
    
        wandb.config.update(args)
  
        testset 												= RelDataset(dev_data, args)
        testloader     							   				= DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)

        model.load_state_dict(torch.load(checkpoint_file))
        model.eval()

        pt, rt, f1t 	 = seen_eval(model, testloader, device=device)
        print(f'Test data {f1t} \t Prec {pt} \t Rec {rt}')
        
        wandb.log({"dev_f1": 		f1t})
        wandb.log({"dev_precision": pt})
        wandb.log({"dev_recall": 	rt})


        testset 												= RelDataset(test_data, args)
        testloader     							   				= DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)
  
        model.load_state_dict(torch.load(checkpoint_file))
        model.eval()

        pt, rt, f1t 	 = seen_eval(model, testloader, device=device)
        print(f'Test data {f1t} \t Prec {pt} \t Rec {rt}')
    
        wandb.log({"test_f1": 		f1t})
        wandb.log({"test_precision": pt})
        wandb.log({"test_recall": 	rt})
        wandb.log({"seed": 		args.seed})
        wandb.log({"dep": 		args.dep})
        wandb.log({"gnn_depth": args.gnn_depth})
        wandb.log({"connection":args.connection})
        wandb.log({"model": 	args.model_name})
        wandb.log({"data": 	    args.dataset})
        wandb.log({'dep_model': args.dep_model})
        wandb.log({"lr": 		args.lr})
        wandb.log({"src_lang": 	args.src_lang})
        wandb.log({"tgt_lang": 	args.tgt_lang})



    if args.mode											== 'predict':

        # checkpoint_file 								    =  f'../ckpts/{args.dataset}/{args.src_lang}_{args.src_lang}-model_{args.model_name}-parser_{args.dep_model}-gnn_{args.gnn_model}-connection_{args.connection}-setting_{args.setting}-gnn-depth_{args.gnn_depth}-seed_{args.seed}'
        checkpoint_file 								    =  f'../ckpts/{file_format}'


        if not check_file(checkpoint_file) :
            print('MODEL CANNOT BE LOADED BY THESE SPECIFICATIONS')
            return

        # wandb.login()
        # wandb.init(project="multilingual_re", entity="flow-graphs-ORG", name=f'{checkpoint_file.split("/")[-1]}')
    
        # wandb.config.update(args)      

        testset 												= RelDataset(test_data, args)
        testloader     							   				= DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)
  
        model.load_state_dict(torch.load(checkpoint_file))
        model.eval()

        
        pred_dict 		 	                                      =   ddict(list)


        results_dict 	                                      =   seen_predict(model, testloader, device=device)
        y_pred, y_true 	                                      =   results_dict['y_pred'], results_dict['y_true']

        assert len(y_pred) == len(y_true) == len(test_data)

        for idx in tqdm(range(0, len(test_data))):
            pred_dict['id'].append(idx)
            pred_dict['doc_text'].append(test_data[idx]['doc_text'])
            pred_dict['true_rel'].append(relation_dict[str(y_true[idx])])
            pred_dict['pred_rel'].append(relation_dict[str(y_pred[idx])])


        
        pred_file 								   				=  f'../predictions/{file_format}.csv'

    
        # dump pickle file
        pred_df 								   				= pd.DataFrame(pred_dict)
        pred_df.to_csv(pred_file, index=False)
        


if __name__ =='__main__':	
    args                            =   get_args()
    main(args)
