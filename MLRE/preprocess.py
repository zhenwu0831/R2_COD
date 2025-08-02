from helper import *
from datasets import load_dataset
import stanza, torch
# from trankit import Pipeline
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel 
import networkx as nx
from karateclub.node_embedding.neighbourhood import Walklets    




def create_indore_el_data():
    from transformers import AutoTokenizer, AutoModel 
    
    rand_vec                             = np.random.randn(128)
 
    import stanza, torch
    from trankit import Pipeline
    from torch_geometric.data import Data

    deprel_dict                         = load_deprels(enhanced=False)
    entity_dict                         = load_pickle(f'{args.rel_dir}/ents.pkl')
    relation_dict                       = load_pickle(f'{args.rel_dir}/rels.pkl')
    ent_embs_dict                       = load_pickle(f'{args.rel_dir}/ent_embs.pkl')

    # languages                          = ['te']
    languages                           = ['te','en','hi']#,'english','bengali','hindi']
    lang_code                             = {'bn':'bengali','en':'english','hi':'hindi','te':'telugu'}
    # lang_code                         = {'bengali':'bn','english':'en','hindi':'hi','telugu':'te'}
    # languages                         = ['bengali']
    lang_sents_path                        = f'{args.rel_dir}/lang_el_sents.dill'
    if os.path.exists(lang_sents_path):
        lines_dict                         = load_dill(lang_sents_path)
    else:
        lines_dict                         = ddict(lambda:ddict(list))
        for lang in tqdm(languages):
            lines                         = open(f'{args.rel_dir}/{lang_code[lang]}_indore.tsv').readlines()
            random.shuffle(lines)
            lines_dict[lang]['train']     = lines[0:int(len(lines)*args.train_ratio)]
            lines_dict[lang]['dev']     = lines[int(len(lines)*args.train_ratio):int(len(lines)*(args.train_ratio+args.dev_ratio))]
            lines_dict[lang]['test']    = lines[int(len(lines)*(args.train_ratio+args.dev_ratio)):]
        dump_dill(lines_dict, lang_sents_path)
        
    if args.bert_model      == 'mbert':
        tokenizer             =     AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model                 =     AutoModel.from_pretrained('bert-base-multilingual-uncased')

    if args.bert_model      == 'xlmr':
        tokenizer             =     AutoTokenizer.from_pretrained('xlm-roberta-base')
        model                 =     AutoModel.from_pretrained('xlm-roberta-base')
    
 
    relation_desc            = {}
    for rel in relation_dict:
        rel_toks             = tokenizer(rel.replace('_',''), return_tensors='pt')
        relation_desc[rel]    = model(**rel_toks)['pooler_output'].cpu().squeeze(dim=1)

    
    for lang in languages:
        data                               = ddict(lambda: ddict(list))
        if lang                            !='bengali': 
            if args.dep_model == 'stanza':
                stanza_nlp                              = stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma,depparse')
            elif args.dep_model == 'trankit':
                trankit_nlp                             = Pipeline(lang_code[lang])
   
        for split in ['train','dev','test']:
            for line in tqdm(lines_dict[lang][split]):
                rel, sent, ent_1, ent_2 = line.strip().split('\t')
                orig_sent                 = sent

                e1_start,   e2_start    = sent.index('<e1>')+4, sent.index('<e2>')+4
                e1_end,     e2_end      = sent.index('</e1>'), sent.index('</e2>')
                e1_span,    e2_span     = sent[e1_start:e1_end], sent[e2_start:e2_end]

                if e1_start > e2_end:
                    e2_start                = sent.index('<e2>')
                    sent                    = sent.replace('<e2>','')
                    e2_end                     = sent.index('</e2>')
                    sent                    = sent.replace('</e2>','')    
                    e1_start                = sent.index('<e1>')
                    sent                    = sent.replace('<e1>','')
                    e1_end                     = sent.index('</e1>')
                    sent                    = sent.replace('</e1>','')
                else:
                    e1_start                = sent.index('<e1>')
                    sent                    = sent.replace('<e1>','')
                    e1_end                     = sent.index('</e1>')
                    sent                    = sent.replace('</e1>','')
                    e2_start                = sent.index('<e2>')
                    sent                    = sent.replace('<e2>','')
                    e2_end                     = sent.index('</e2>')
                    sent                    = sent.replace('</e2>','')    
                
                assert e1_span == sent[e1_start:e1_end] and e2_span == sent[e2_start:e2_end]

                # sent_toks                                     = tokenizer(sent, return_offsets_mapping=True, max_length=512,add_special_tokens=False)
                sent_toks                                         = tokenizer(sent, return_offsets_mapping=True, max_length=args.max_seq_len)
                mlm_toks                                         = sent_toks['input_ids']
                tok_range                                         = sent_toks['offset_mapping']
                e1_toks                                            = [0]+[1 if elem[0]>=e1_start and elem[1]<=e1_end else 0 for elem in tok_range[1:-1]]+[0]
                e2_toks                                         = [0]+[1 if elem[0]>=e2_start and elem[1]<=e2_end else 0 for elem in tok_range[1:-1]]+[0]
                e1_type, e2_type                                 = np.zeros(len(entity_dict)), np.zeros(len(entity_dict))
                e1_type[entity_dict[ent_1]]                     = 1 
                e2_type[entity_dict[ent_2]]                     = 1 
                rel_type                                         = np.zeros(len(relation_dict))
                rel_type[relation_dict[rel]]                    = 1
    
                node_dict                                         = {}
                node_idx_dict                                     = {}
                node_mask_dict                                     = {}
                edge_arr                                         = []
                dep_arr                                         = []
    
                # Specifically for the root that is attached to the main verb
                # STAR NODE
                node_dict[(-1,-1)]                                = 0
                # node_idx_dict[(-1,-1)]                        = (0,len(mlm_toks))
                node_idx_dict[(-1,-1)]                            = (1,len(mlm_toks)-1)
                node_mask_dict[(-1,-1)]                            = 0
    
                
                if lang == 'bn': continue;
                    # x, edge_index, edge_type, n1_mask, n2_mask        = torch.FloatTensor(np.array(x)), torch.LongTensor(edge_index), torch.LongTensor(edge_type), torch.LongTensor(n1_mask),torch.LongTensor(n2_mask)
                    # dep_data                                         = Data(x=x, edge_index= edge_index, edge_type=edge_type, n1_mask=n1_mask, n2_mask=n2_mask)
                else:                
                    if args.dep_model == 'stanza':
                        dep_doc                                         = stanza_nlp(sent)    
                        num_sents                                         = len(dep_doc.sentences)
            
                        for sent_cnt, dep_sent in enumerate(dep_doc.sentences):
                            for word in dep_sent.words:
                                if         word.start_char    >= e1_start and word.end_char <= e1_end: dep_val = 1
                                elif    word.start_char >= e2_start and word.end_char <= e2_end: dep_val = 2
                                else:   dep_val = 0
                                dep_arr.append(((sent_cnt, word.id), word.text, (sent_cnt, word.head), word.deprel, word.start_char, word.end_char, dep_val))
                    elif args.dep_model == 'trankit':
                        dep_doc                                         = trankit_nlp(sent)    
                        num_sents                                         = len(dep_doc["sentences"])
            
                        for sent_cnt, dep_sent in enumerate(dep_doc["sentences"]):
                            for word in dep_sent["tokens"]:
                                start_char, end_char = word["dspan"]
                                if         start_char    >= e1_start and end_char <= e1_end: dep_val = 1
                                elif    start_char >= e2_start and end_char <= e2_end: dep_val = 2
                                else:   dep_val = 0
                                dep_arr.append(((sent_cnt, word["id"]), word["text"], (sent_cnt, word["head"]), word["deprel"], start_char, end_char, dep_val))
                    start_tok_idx = 1; end_tok_idx =1;
     
                    for elem in dep_arr:
                        start_idx, word, end_idx, deprel, start_char, end_char, mask_val  = elem
                        if start_idx not in node_dict        :node_dict[start_idx]= len(node_dict)
                        if end_idx   not in node_dict        :node_dict[end_idx]  = len(node_dict)
      
                        # for idx in range(start_tok_idx, len(tok_range)):
                        for idx in range(start_tok_idx, len(tok_range)):
                            curr_start, curr_end                 = tok_range[idx][0], tok_range[idx][1]
                            if         curr_end     == 0  and idx ==len(tok_range)-1 : end_tok_idx =idx; break
                            elif     curr_end     <= start_char    : start_tok_idx = idx +1; continue
                            elif     curr_end    <= end_char        : continue
                            elif    curr_start     >= end_char        : end_tok_idx = idx; break
                            
                        # if     idx == len(tok_range) -2        : end_tok_idx = idx+1
                        # if     idx == len(tok_range) -1        : end_tok_idx = idx+1
      
                        node_idx_dict[start_idx]            = (start_tok_idx, end_tok_idx)
                        if ':' in deprel:
                            deprel                             = deprel.split(':')[0] 
                        # edge_index[0].append(start_idx)
                        # edge_index[1].append(end_idx)
                        # edge_type.append(deprel_dict[deprel])
                        edge_arr.append((start_idx, end_idx,deprel_dict[deprel]))
                        node_mask_dict[start_idx]        = mask_val
                        start_tok_idx                      = end_tok_idx
      
                    ################ CREATED A GLOBAL GRAPH ACROSS ALL SENTENCES #######################

                    for sent_num in range(num_sents):
                        tok_idxs             = [node_idx_dict[elem] for elem in node_idx_dict if elem[0]== sent_num]
                        min_tok_idx         = min([tok_idx[0] for tok_idx in tok_idxs])
                        max_tok_idx         = max([tok_idx[1] for tok_idx in tok_idxs])

                        node_idx_dict[(sent_num,0)]                    = (min_tok_idx, max_tok_idx)
                        node_mask_dict[(sent_num,0)]                = 0
      
                    x, edge_index, edge_type, n1_mask, n2_mask    = [],[[],[]],[],[],[]
                    for node in node_dict:
                        six, eix                                     = node_idx_dict[node]
                        temp_ones                                     = torch.ones((512,))*-torch.inf
      
                        if six < eix    : temp_ones[six:eix]=0
                        elif six == eix : temp_ones[six]    =0
                        else: import pdb; pdb.set_trace()
                        x.append(temp_ones)
      
                        mask = node_mask_dict[node]
                        if mask == 0: n1_mask.append(0); n2_mask.append(0)
                        if mask == 1: n1_mask.append(1); n2_mask.append(0)
                        if mask == 2: n1_mask.append(0); n2_mask.append(1)

                    for edge in edge_arr:
                        n1, n2, rel_idx                             = edge
                        edge_index[0].append(node_dict[n1])
                        edge_index[1].append(node_dict[n2])
                        edge_type.append(rel_idx)
      
                    for sent_num in range(num_sents):
                        edge_index[0].append(node_dict[(sent_num,0)])
                        edge_index[1].append(node_dict[(-1,-1)])
                        edge_type.append(deprel_dict['STAR'])
      
                    try:
                        x, edge_index, edge_type, n1_mask, n2_mask        = torch.stack(x, dim=0), torch.LongTensor(edge_index), torch.LongTensor(edge_type), torch.LongTensor(n1_mask),torch.LongTensor(n2_mask)
                        dep_data                                         = Data(x=x, edge_index= edge_index, edge_type=edge_type, n1_mask=n1_mask, n2_mask=n2_mask)
                    except Exception as e:
                        import pdb; pdb.set_trace()

                if orig_sent in ent_embs_dict[lang]:
                    ent1_emb                                         = ent_embs_dict[lang][orig_sent][0]
                    ent2_emb                                         = ent_embs_dict[lang][orig_sent][1]
                    if isinstance(ent1_emb, int)                    : ent1_emb = rand_vec
                    if isinstance(ent2_emb, int)                     : ent2_emb = rand_vec
                else:
                    ent1_emb = rand_vec; ent2_emb = rand_vec                    

                data[split]['rels'].append({
                    'tokens'    : mlm_toks,
                    'tok_range'    : tok_range,    
                    'arg1_ids'    : e1_toks,
                    'arg2_ids'    : e2_toks,
                    'arg1_type' : e1_type,
                    'arg2_type' : e2_type,
                    'desc_emb'    : relation_desc[rel],
                    'label'        : rel_type,
                    'dep_data'    : dep_data,
                    'sent'        : sent,
                    'orig_sent'    : orig_sent, 
                    'arg1_emb'    : ent1_emb,
                    'arg2_emb'    : ent2_emb
                })
        dump_dill(data, f'{args.rel_dir}/{lang}_el_rels.dill')


def get_dep_arr(doc_text, dep_nlp, e1_start, e1_end, e2_start, e2_end):

    dep_arr            = []

    if args.dep_model == 'stanza':
        dep_doc                                        = dep_nlp(doc_text)    
        num_sents                                      = len(dep_doc.sentences)

        for sent_cnt, dep_sent in enumerate(dep_doc.sentences):
            
            words                = []
            start_char, end_char = 0, 0

            for word in dep_sent.words:

                if  word.start_char is not None and word.end_char is not None:
                    start_char, end_char = word.start_char, word.end_char

                if      start_char >= e1_start and end_char <= e1_end: dep_val = 1

                elif    start_char >= e2_start and end_char <= e2_end: dep_val = 2

                else:   dep_val = 0

                if  word.deprel is not None:
                    dep_arr.append(((sent_cnt, word.id), word.text, (sent_cnt, word.head), word.deprel, start_char, end_char, dep_val))

    elif args.dep_model == 'trankit':
        dep_doc                                             = dep_nlp(doc_text)    
        num_sents                                           = len(dep_doc["sentences"])

        for sent_cnt, dep_sent in enumerate(dep_doc["sentences"]):
            for word in dep_sent["tokens"]:
                
                start_char, end_char    = word["dspan"]
                if      start_char      >= e1_start and end_char <= e1_end: dep_val = 1
                elif    start_char      >= e2_start and end_char <= e2_end: dep_val = 2

                else:   dep_val = 0

                if 'expanded' in word:                                        
                    for elem in word['expanded']:
                        dep_arr.append(((sent_cnt, elem["id"]), elem["text"], (sent_cnt, elem["head"]), elem["deprel"], start_char, end_char, dep_val))

                else:
                    dep_arr.append(((sent_cnt, word["id"]), word["text"], (sent_cnt, word["head"]), word["deprel"], start_char, end_char, dep_val))

    return dep_arr, num_sents


def create_redfm_walklet():

    ds = load_dataset("Babelscape/REDFM", "all_languages")

    '''
    train: Dataset({
        features: ['docid', 'title', 'uri', 'lan', 'text', 'entities', 'relations'],
        num_rows: 10337
    })
    '''

    if args.lang in ['ar', 'zh']:
        splits = ['validation', 'test']
    else:
        splits = ['train', 'validation', 'test']

    model_dict = {
        'mbert-base'     : 'bert-base-multilingual-uncased',
        'xlmr-base'      : 'xlm-roberta-base',
        'infoxlm-base'   : 'microsoft/infoxlm-base',
        'rembert'        : 'google/rembert',
        'xlmr-large'     : 'xlm-roberta-large',
        'infoxlm-large'  : 'microsoft/infoxlm-large',
    }

    tokenizer     = AutoTokenizer.from_pretrained(model_dict[args.ml_model])
    model         = AutoModel.from_pretrained(model_dict[args.ml_model])

    relation_dict = ddict(int)
    lang_dict     = ddict(int)

    for split in splits:
        data = ds[split]
        for elem in data:
            rels = elem['relations']
            for rel in rels:
                relation_dict[rel['predicate']] += 1

            lang_dict[elem['lan']] += 1    
    
    deprel_dict                                                    = load_deprels(enhanced=False)
    lang_code_dict                                                 = {'bn':'bengali','en':'english','hi':'hindi','te':'telugu', 'ar': 'arabic', 'de': 'german', 'es': 'spanish', 'fr': 'french', 'it': 'italian', 'ja': 'japanese', 'ko': 'korean', 'pt': 'portuguese', 'ru': 'russian', 'zh': 'chinese'}
    
    # code to check if a given file exists or not


    if os.path.exists(f'{args.rel_dir}/redfm/{args.lang}_{args.ml_model}_{args.dep_model}_combined.dill'):
        print("File exists")
        exit()    

    splitwise_data                                                = ddict(list)      
    lang                                                          = args.lang                 

    if args.dep_model == 'stanza':
        dep_nlp      = stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma,depparse', use_gpu=True)

    elif args.dep_model == 'trankit':
        dep_nlp     = Pipeline(lang_code_dict[lang], gpu=False)

    for split in splits:
        data                                                      = ds[split].filter(lambda x: x['lan'] == lang)

        for curr_data in tqdm(data, desc=f'{lang}_{split}'):
            rels                                                  = curr_data['relations']
            doc_text                                              = curr_data['text']

            for rel in rels:
                try:
                    predicate                                         = rel['predicate']
                    e1, e2                                            = rel['subject'], rel['object']
                    
                    e1_start, e1_end                                  = e1['start'], e1['end']
                    e2_start, e2_end                                  = e2['start'], e2['end']


                    sent_toks                                         = tokenizer(doc_text, return_offsets_mapping=True, max_length=args.max_seq_len)

                    mlm_toks                                          = sent_toks['input_ids']
                    tok_range                                         = sent_toks['offset_mapping']

                    e1_toks                                           = [0]+[1 if elem[0]>=e1_start and elem[1]<=e1_end else 0 for elem in tok_range[1:-1]]+[0]
                    e2_toks                                           = [0]+[1 if elem[0]>=e2_start and elem[1]<=e2_end else 0 for elem in tok_range[1:-1]]+[0]

                    rel_type                                          = np.zeros(len(relation_dict))
                    rel_type[predicate]                               = 1
    
                    '''
                    Create the dependency array for the current sentence
                    '''

                    dep_arr, num_sents                                = get_dep_arr(doc_text, dep_nlp, e1_start, e1_end, e2_start, e2_end)
                    dep_data, node2idx_dict                           = get_embeddings(dep_arr, num_sents, deprel_dict, mlm_toks, tok_range, parser_type=args.dep_model)

                    # import pdb; pdb.set_trace()

                    splitwise_data[split].append({
                        'tokens'      : mlm_toks,
                        'tok_range'   : tok_range,    
                        'e1_ids'      : e1_toks,
                        'e2_ids'      : e2_toks,
                        'label'       : rel_type,
                        'dep_data'    : dep_data,
                        'doc_text'    : doc_text,
                        'node2idx_dict' : node2idx_dict,
                    })
                
                except Exception as e:
                    print(f'Error: {e}')
                    continue
                                                
    dump_dill(splitwise_data, f'{args.rel_dir}/redfm/{args.lang}_{args.ml_model}_{args.dep_model}_combined.dill')        


def get_embeddings(dep_arr, num_sents, dep_dict, mlm_toks, tok_range, parser_type='stanza'):

    # modify this code to get the dependency arr 

    node_dict = {}

    node_dict                                           = {}
    node_mask_dict                                      = {}
    node_range_dict                                     = {}

    edges                                               = set()
    # STAR NODE
    node_dict[(-1,-1)]                                  = 0
    node_mask_dict[(-1,-1)]                             = 0
    # we are also going to store the range of the nodes according to the corresponding token positions 
    node_range_dict[(-1,-1)]                            = (0,len(mlm_toks))


    edge_index                                          = [[],[]]
    edge_type                                           = []

    start_tok_idx = 1; end_tok_idx =1
    # ((0, 1), 'From', (0, 5), 'case', 0, 4, 0)
    # (sent_cnt, word["id"]), word["text"], (sent_cnt, word["head"]), word["deprel"], start_char, end_char, dep_val)

    for item in dep_arr:

        node_idx, word, head_idx, deprel, start_char, end_char, mask_val  = item

        if node_idx not in node_dict:
            node_dict[node_idx] = len(node_dict)
        if head_idx not in node_dict:
            node_dict[head_idx] = len(node_dict)

        if parser_type == 'stanza':
            if ':' in deprel:
                deprel                             = deprel.split(':')[0] 

        node_mask_dict[node_idx]                   = mask_val

        '''
        Aligning the current element in the dependency array with the token range
        '''
        for idx in range(start_tok_idx, len(tok_range)):            
            ## aligning the current elemnet in the dependency array with the token range
            curr_start, curr_end  = tok_range[idx][0], tok_range[idx][1]

            if   curr_end       == 0  and idx ==len(tok_range)-1 : 
                end_tok_idx =idx
                break

            elif curr_end       <= start_char: 
                start_tok_idx = idx +1
                continue

            elif curr_end       <= end_char  :
                continue

            elif curr_start     >= end_char  :
                end_tok_idx = idx
                break
        
        node_range_dict[node_idx]             = (start_tok_idx, end_tok_idx)

        edges.add((node_dict[node_idx], node_dict[head_idx], dep_dict[deprel]))        


    ###  ADDING THE INFORMATION CORRESPONDING TO THE INDIVIDUAL SENTENCE NODES, i.e. their range and mask values ####
    for sent_num in range(num_sents):
        node_mask_dict[(sent_num,0)]                = 0
        if (sent_num,0) not in node_dict:
            node_dict[(sent_num,0)]                 = len(node_dict)

        edges.add((node_dict[(sent_num,0)], node_dict[(-1,-1)], dep_dict['STAR']))
    
        tok_idxs                                    = [node_range_dict[elem] for elem in node_range_dict if elem[0]== sent_num]
        min_tok_idx                                 = min([tok_idx[0] for tok_idx in tok_idxs])
        max_tok_idx                                 = max([tok_idx[1] for tok_idx in tok_idxs])
        node_range_dict[(sent_num,0)]               = (min_tok_idx, max_tok_idx)
    
    #### Computing the walklet embeddings ####

    node2idx_dict                                   = {}
    for node in node_dict:
        node2idx_dict[node_dict[node]]              = node
    
    node_list       = list(node2idx_dict.keys())
    edge_list       = [(edge[0], edge[1]) for edge in edges]

    G = nx.DiGraph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    model = Walklets(dimensions=192)
    model.fit(G)
    embedding = model.get_embedding()


    ### Computing now the token range values  ####

    x, tok_x,  edge_index, edge_type, n1_mask, n2_mask    = [], [], [[],[]],[],[],[]

    for nix in node2idx_dict:
        x.append(embedding[nix])

        node = node2idx_dict[nix]
        mask = node_mask_dict[node]

        tok_range = node_range_dict[node]
        six, eix  = tok_range

        temp_ones                                   = torch.ones((args.max_seq_len,))*-torch.inf

        if six      < eix  :
            temp_ones[six:eix]        = 0
        elif six    == eix :
            temp_ones[six]            = 0

        tok_x.append(temp_ones)

        if mask == 0: n1_mask.append(0); n2_mask.append(0)
        if mask == 1: n1_mask.append(1); n2_mask.append(0)
        if mask == 2: n1_mask.append(0); n2_mask.append(1)

    for edge in edges:
        n1, n2, rel_idx                             = edge
        edge_index[0].append(n1)
        edge_index[1].append(n2)
        edge_type.append(rel_idx)


    # x, tok_x, edge_index, edge_type, n1_mask, n2_mask       = torch.Tensor(x), torch.Tensor(tok_x), torch.LongTensor(edge_index), torch.LongTensor(edge_type), torch.LongTensor(n1_mask),torch.LongTensor(n2_mask)

    x                                                       = torch.Tensor(np.array(x))
    tok_x                                                   = torch.stack(tok_x, dim=0)
    edge_index                                              = torch.LongTensor(edge_index)
    edge_type                                               = torch.LongTensor(edge_type)
    n1_mask                                                 = torch.LongTensor(n1_mask)
    n2_mask                                                 = torch.LongTensor(n2_mask)
    
    dep_data                                                = Data(x=x, tok_x = tok_x, edge_index= edge_index, edge_type=edge_type, n1_mask=n1_mask, n2_mask=n2_mask)

    return dep_data, node2idx_dict 
    
            
            



def create_indore_walklet():
    model_dict = {
        'mbert-base'     : 'bert-base-multilingual-uncased',
        'xlmr-base'      : 'xlm-roberta-base',
        'infoxlm-base'   : 'microsoft/infoxlm-base',
        'rembert'        : 'google/rembert',
        'xlmr-large'     : 'xlm-roberta-large',
        'infoxlm-large'  : 'microsoft/infoxlm-large',
    }

    tokenizer     = AutoTokenizer.from_pretrained(model_dict[args.ml_model])
    model         = AutoModel.from_pretrained(model_dict[args.ml_model])


    relation_dict = ddict(int)
    lang_dict     = ddict(int)

    
    deprel_dict                                                    = load_deprels(enhanced=False)
    lang_code_dict                                                 = {'bn':'bengali','en':'english','hi':'hindi','te':'telugu', 'ar': 'arabic', 'de': 'german', 'es': 'spanish', 'fr': 'french', 'it': 'italian', 'ja': 'japanese', 'ko': 'korean', 'pt': 'portuguese', 'ru': 'russian', 'zh': 'chinese'}
    
    # code to check if a given file exists or not

    if os.path.exists(f'{args.rel_dir}/indore/{args.lang}_{args.ml_model}_{args.dep_model}.dill'):
        print("File exists")
        exit()    

    splitwise_data                                                = ddict(list)                  
    
    indore_dir = '/data/shire/data/NO-BACKUP/multilingual_KGQA/IndoRE/data'

    indore_data = ddict(lambda: ddict(list))

    lang_sents_path                        = f'../data/indore/dataset.dill'
    if os.path.exists(lang_sents_path):
        lines_dict                         = load_dill(lang_sents_path)
    else:
        lines_dict                         = ddict(lambda:ddict(list))
        for lang in ['en','hi','te']:
            lines                          = open(f'{indore_dir}/{lang_code_dict[lang]}_indore.tsv').readlines()
            random.shuffle(lines)
            lines_dict[lang]['train']      = lines[0:int(len(lines)*args.train_ratio)]
            lines_dict[lang]['validation']  = lines[int(len(lines)*args.train_ratio):int(len(lines)*(args.train_ratio+args.dev_ratio))]
            lines_dict[lang]['test']       = lines[int(len(lines)*(args.train_ratio+args.dev_ratio)):]

        dump_dill(lines_dict, lang_sents_path)
    
    
    lang_data = load_dill(lang_sents_path)[args.lang]

    if args.dep_model == 'stanza':
        dep_nlp      = stanza.Pipeline(lang=args.lang, processors='tokenize,pos,lemma,depparse', use_gpu=False)

    elif args.dep_model == 'trankit':
        dep_nlp     = Pipeline(lang_code_dict[args.lang], gpu=False)

    deprel_dict                         = load_deprels(enhanced=False)
    entity_dict                         = load_pickle(f'{indore_dir}/ents.pkl')
    relation_dict                       = load_pickle(f'{indore_dir}/rels.pkl')

    lang                                = args.lang


    for split in ['train','validation','test']:

        data                                                      = lang_data[split]

        
        for line in tqdm(data, desc=f'{lang}_{split}'):

            try:

                rel, sent, ent_1, ent_2     = line.strip().split('\t')
                orig_sent                   = sent

                e1_start,   e2_start        = sent.index('<e1>')+4, sent.index('<e2>')+4
                e1_end,     e2_end          = sent.index('</e1>'), sent.index('</e2>')
                e1_span,    e2_span         = sent[e1_start:e1_end], sent[e2_start:e2_end]

                if e1_start > e2_end:
                    e2_start                = sent.index('<e2>')
                    sent                    = sent.replace('<e2>','')
                    e2_end                  = sent.index('</e2>')
                    sent                    = sent.replace('</e2>','')    
                    e1_start                = sent.index('<e1>')
                    sent                    = sent.replace('<e1>','')
                    e1_end                  = sent.index('</e1>')
                    sent                    = sent.replace('</e1>','')
                else:
                    e1_start                = sent.index('<e1>')
                    sent                    = sent.replace('<e1>','')
                    e1_end                  = sent.index('</e1>')
                    sent                    = sent.replace('</e1>','')
                    e2_start                = sent.index('<e2>')
                    sent                    = sent.replace('<e2>','')
                    e2_end                  = sent.index('</e2>')
                    sent                    = sent.replace('</e2>','')    
                
                assert e1_span == sent[e1_start:e1_end] and e2_span == sent[e2_start:e2_end]

                sent_toks                                           = tokenizer(sent, return_offsets_mapping=True, max_length=args.max_seq_len)
                mlm_toks                                            = sent_toks['input_ids']
                tok_range                                           = sent_toks['offset_mapping']
                e1_toks                                             = [0]+[1 if elem[0]>=e1_start and elem[1]<=e1_end else 0 for elem in tok_range[1:-1]]+[0]
                e2_toks                                             = [0]+[1 if elem[0]>=e2_start and elem[1]<=e2_end else 0 for elem in tok_range[1:-1]]+[0]
                
                rel_type                                            = np.zeros(len(relation_dict))
                rel_type[relation_dict[rel]]                        = 1

                '''
                Create the dependency array for the current sentence
                '''

                dep_arr, num_sents                                = get_dep_arr(sent, dep_nlp, e1_start, e1_end, e2_start, e2_end)

                dep_data, node2idx_dict                           = get_walklet_embeddings(dep_arr, num_sents, deprel_dict, parser_type=args.dep_model)

                splitwise_data[split].append({
                        'tokens'      : mlm_toks,
                        'tok_range'   : tok_range,    
                        'e1_ids'      : e1_toks,
                        'e2_ids'      : e2_toks,
                        'label'       : rel_type,
                        'dep_data'    : dep_data,
                        'doc_text'    : sent,
                        'node2idx_dict' : node2idx_dict,                    
                    })
            except Exception as e:
                print(f'Error: {e}')
                continue
                
                
        dump_dill(splitwise_data, f'{args.rel_dir}/indore/{args.lang}_{args.ml_model}_{args.dep_model}_walklet.dill')        


def create_data_config():

    model_names = {
        'mbert-base'     : 'bert-base-multilingual-uncased',
        # 'xlmr-base'      : 'xlm-roberta-base',
        # 'infoxlm-base'   : 'microsoft/infoxlm-base',
        # 'rembert'        : 'google/rembert',
        # 'xlmr-large'     : 'xlm-roberta-large',
        # 'infoxlm-large'  : 'microsoft/infoxlm-large',
    }

    config_dict     = ddict(list)
    cnt             = 1

    for lang in ['it', 'es', 'en', 'de',  'fr', 'zh', 'ar']:
        for dep_model in ['stanza']:
            for ml_model in model_names:
                config_dict['ArrayTaskID'].append(cnt)
                config_dict['MLM'].append(ml_model)
                config_dict['DEP_MODEL'].append(dep_model)

                if 'large' in ml_model:
                    config_dict['MAX_LEN'].append(1024)
                else:
                    config_dict['MAX_LEN'].append(512)

                cnt +=1
                config_dict['LANG'].append(lang)

        config_df = pd.DataFrame(config_dict)
        config_df.to_csv('../configs/redfm_creation_config.csv', index=False, sep=' ')


    config_dict     = ddict(list)
    cnt             = 1

    for lang in ['hi', 'en', 'te']:
        for dep_model in ['stanza']:
            for ml_model in model_names:
                config_dict['ArrayTaskID'].append(cnt)
                config_dict['MLM'].append(ml_model)
                config_dict['DEP_MODEL'].append(dep_model)

                if 'large' in ml_model:
                    config_dict['MAX_LEN'].append(1024)
                else:
                    config_dict['MAX_LEN'].append(512)
                cnt +=1
                config_dict['LANG'].append(lang)

        config_df = pd.DataFrame(config_dict)
        config_df.to_csv('../configs/indore_creation_config.csv', index=False, sep=' ')


def create_train_config():

    model_names = {
        'mbert-base'     : 'bert-base-multilingual-uncased',
        # 'xlmr-base'      : 'xlm-roberta-base',
        # 'infoxlm-base'   : 'microsoft/infoxlm-base',
        # 'rembert'        : 'google/rembert',
        # 'xlmr-large'     : 'xlm-roberta-large',
        # 'infoxlm-large'  : 'microsoft/infoxlm-large',
    }

    config_dict     = ddict(list)
    cnt             = 1

    for dataset in ['redfm']:
        for seed in [15123, 98105, 11737]:
            for lang in ['it',  'es', 'en', 'de', 'fr']:
                    for ml_model in model_names:
                        for setting in ['both']:
                            config_dict['ArrayTaskID'].append(cnt)
                            config_dict['DATASET'].append(dataset)
                            config_dict['MLM'].append(ml_model)
                            config_dict['DEP_MODEL'].append('stanza')
                            config_dict['SRC_LANG'].append(lang)
                            config_dict['TGT_LANG'].append(lang)
                            config_dict['SEED'].append(seed)
                            config_dict['GNN_MODEL'].append('rgcn')
                            config_dict['CONNECTION'].append('residual')
                            config_dict['SETTING'].append(setting)            
                            cnt +=1
                        

    config_df = pd.DataFrame(config_dict)
    config_df.to_csv('../configs/mulco_redfm_train_config.csv', index=False, sep=' ')
    
    config_dict     = ddict(list)
    cnt             = 1

    for dataset in ['indore']:
        for seed in [15123, 98105, 11737]:
            for lang in ['en', 'hi', 'te']:
                    for ml_model in model_names:
                        for connection in ['mulco', 'mulco_doc']:
                            config_dict['ArrayTaskID'].append(cnt)
                            config_dict['DATASET'].append(dataset)
                            config_dict['MLM'].append(ml_model)
                            config_dict['DEP_MODEL'].append('stanza')
                            config_dict['SRC_LANG'].append(lang)
                            config_dict['TGT_LANG'].append(lang)
                            config_dict['SEED'].append(seed)
                            config_dict['GNN_MODEL'].append('rgcn')
                            config_dict['CONNECTION'].append(connection)
                            config_dict['SETTING'].append('both')            
                            cnt +=1
                        

    config_df = pd.DataFrame(config_dict)
    config_df.to_csv('../configs/mulco_indore_train_config.csv', index=False, sep=' ')




def create_test_config():

    model_names = {
        'mbert-base'     : 'bert-base-multilingual-uncased',
        # 'xlmr-base'      : 'xlm-roberta-base',
        # 'infoxlm-base'   : 'microsoft/infoxlm-base',
        # 'rembert'        : 'google/rembert',
        # 'xlmr-large'     : 'xlm-roberta-large',
        # 'infoxlm-large'  : 'microsoft/infoxlm-large',
    }

    config_dict     = ddict(list)
    cnt             = 1

    for dataset in ['redfm']:
        for seed in [15123, 98105, 11737]:
            for src_lang in ['it',  'es', 'en', 'de', 'fr']:
                for tgt_lang in ['it',  'es', 'en', 'de', 'fr', 'zh', 'ar']:
                    for ml_model in model_names:
                        for connection in ['mulco', 'mulco_doc']:
                            config_dict['ArrayTaskID'].append(cnt)
                            config_dict['DATASET'].append(dataset)
                            config_dict['MLM'].append(ml_model)
                            config_dict['DEP_MODEL'].append('stanza')
                            config_dict['SRC_LANG'].append(src_lang)
                            config_dict['TGT_LANG'].append(tgt_lang)
                            config_dict['SEED'].append(seed)
                            config_dict['GNN_MODEL'].append('rgcn')
                            config_dict['CONNECTION'].append(connection)
                            config_dict['SETTING'].append('both')            
                            cnt +=1
                        
                        for setting in ['text']:
                            config_dict['ArrayTaskID'].append(cnt)
                            config_dict['DATASET'].append(dataset)
                            config_dict['MLM'].append(ml_model)
                            config_dict['DEP_MODEL'].append('stanza')
                            config_dict['SRC_LANG'].append(src_lang)
                            config_dict['TGT_LANG'].append(tgt_lang)
                            config_dict['SEED'].append(seed)
                            config_dict['GNN_MODEL'].append('rgcn')
                            config_dict['CONNECTION'].append('mulco')
                            config_dict['SETTING'].append(setting)            
                            cnt +=1




    config_df = pd.DataFrame(config_dict)
    config_df.to_csv('../configs/mulco_redfm_test_config.csv', index=False, sep=' ')
    
    # config_dict     = ddict(list)
    # cnt             = 1

    # for dataset in ['indore']:
    #     for seed in [15123, 98105, 11737]:
    #         for lang in ['en', 'hi', 'te']:
    #                 for ml_model in model_names:
    #                     for connection in ['mulco', 'mulco_doc']:
    #                         config_dict['ArrayTaskID'].append(cnt)
    #                         config_dict['DATASET'].append(dataset)
    #                         config_dict['MLM'].append(ml_model)
    #                         config_dict['DEP_MODEL'].append('stanza')
    #                         config_dict['SRC_LANG'].append(lang)
    #                         config_dict['TGT_LANG'].append(lang)
    #                         config_dict['SEED'].append(seed)
    #                         config_dict['GNN_MODEL'].append('rgcn')
    #                         config_dict['CONNECTION'].append(connection)
    #                         config_dict['SETTING'].append('both')            
    #                         cnt +=1
                        

    # config_df = pd.DataFrame(config_dict)
    # config_df.to_csv('../configs/mulco_indore_train_config.csv', index=False, sep=' ')



if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Arguments for analysis')
    parser.add_argument('--step',           type=str, default='create_redfm', help='Process')
    parser.add_argument('--dep_model',      type=str, default='stanza', help='Dependency Parser')
    parser.add_argument('--ml_model',       type=str, default='mbert-base', help='Choice of Multilingual model')
    parser.add_argument('--max_seq_len',    type=int, default=512, help='Max sequence length')
    parser.add_argument('--rel_dir',        type=str, default='../data', help='Dump dataset')
    parser.add_argument('--lang',           type=str, default='en', help='Language to process')
    parser.add_argument('--train_ratio',    type=float, default=0.8, help='Train ratio')
    parser.add_argument('--dev_ratio',      type=float, default=0.1, help='Dev ratio')
    parser.add_argument('--dataset',        type=str, default='redfm', help='Dataset to process')

    args = parser.parse_args()

    print(args)

    # indore_dir = '/data/shire/data/NO-BACKUP/multilingual_KGQA/IndoRE/data'
    # relation_dict                       = load_pickle(f'{indore_dir}/rels.pkl')
    # inv_relation_dict                   = {v:k for k,v in relation_dict.items()}
    
    # with open(f'../data/indore/relation_dict.json', 'w') as f:
    #     json.dump(inv_relation_dict, f, indent=4)


    if args.step == 'create_redfm':
        create_redfm_walklet()

    if args.step == 'create_indore':
        create_indore_walklet()

    if args.step == 'check_tok':
        check_tokenizers()    

    if args.step == 'create_data_config':
        create_data_config()
    
    if args.step == 'create_train_config':
        create_train_config()
    
    if args.step == 'create_redfm_prompt':
        create_redfm_prompt_data()
    
    if args.step == 'create_indore_prompt':
        create_indore_prompt_data()
    
    if args.step == 'create_test_config':
        create_test_config()
    
    if args.step == 'combine':
        combine_graph_embeddings()