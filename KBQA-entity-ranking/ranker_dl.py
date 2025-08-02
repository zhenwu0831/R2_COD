from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence
import torch
from copy import deepcopy
from tqdm import tqdm
import os

def compute_global_label_map(train_data, dev_data, test_data):
    """
    Computes a global label map from train, dev, and test sets.
    Ensures labels are consistently mapped across all splits.
    """
    unique_labels = sorted(set(ele["isomorphism"] for ele in (train_data + dev_data + test_data)))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    return label_map, len(label_map)

class UnifiedDataset(Dataset):
    def __init__(self, dataset, params, tokenizer):
        self.dataset = self.preprocess_dataset(dataset)
        self.params = params
        self.tokenizer = tokenizer
        self.dataset = self.tokenize_dataset()

    def __len__(self):
        return len(self.dataset)

    def map_entities(self, struct_in, node_dict):
        for entity, index in node_dict.items():
            struct_in = struct_in.replace(entity, f"<E{index}>")
        return struct_in

    def tokenize_dataset(self):
        """
        Pre-tokenize the dataset and extract entity spans.
        """
        tokenized_data = []
        for ele in tqdm(self.dataset, desc="Tokenizing dataset"):
            output_answers = ele["answers"]
            ID = ele["ID"]
            question = ele["question"]
            struct_in = ele["struct_in"]
            iso = ele["isomorphism"]

            struct_in = self.map_entities(struct_in, ele['node_data']['node_dict']) 

            
            seq_in = f"{ele['text_in']} ; structured knowledge: {struct_in}"
            tokenized_input = self.tokenizer(
                seq_in, padding="max_length", truncation=True, max_length=self.params.input_max_length
            )
            
            entity_spans = self.get_entity_token_indices(
                tokenized_input["input_ids"], self.get_entities_from_struct_in(struct_in), self.tokenizer
            )

            # check that len(node_dict) = len(entity_spans)
            if not len(ele['node_data']['node_dict']) == len(entity_spans):
                import pdb; pdb.set_trace()  # Debugging line to inspect variables

            # process output_answers such that if it starts with "ns:", it is removed
            output_answers = [ans[0][3:] if ans[0].startswith("ns:") else ans[0] for ans in output_answers]
            output_answers = [self.map_entities(ans, ele['node_data']['node_dict']) for ans in output_answers]  # Map answers to node indices

            # Create labels for BCE loss (1 if entity is in answers, 0 otherwise)
            entity_labels = {entity: int(entity in output_answers) for entity in entity_spans.keys()}

            # import pdb; pdb.set_trace() 
            # Sort keys like <E0>, <E1>, ... based on the number
            def sort_e_key(k):
                return int(k[2:-1]) if k.startswith("<E") and k.endswith(">") else float("inf")

            sorted_keys = sorted(entity_spans.keys(), key=sort_e_key)

            entity_spans_sorted = {k: entity_spans[k] for k in sorted_keys}
            entity_labels_sorted = {k: entity_labels.get(k, 0) for k in sorted_keys}

            # import pdb; pdb.set_trace() 
            # Graph view
            tokens = self.tokenizer(
                question, padding="max_length", truncation=True, max_length=self.params.max_seq_len
            )

            graph_view = {
                'input_ids': torch.LongTensor(tokens["input_ids"]),
                'attention_mask': torch.LongTensor(tokens["attention_mask"]),
                'gnn_data': Data(
                    x=torch.tensor(ele['node_data']['x']),
                    edge_index=torch.tensor(ele['node_data']['edge_index']),
                    edge_type=torch.tensor(ele['node_data']['edge_type']),
                    answer_x=torch.tensor(ele['node_data']['answer_x']),
                    head_x=torch.tensor(ele['node_data']['head_x']),
                    # edge_label=torch.tensor(ele['node_data']['edge_label'])
                ),
                'answers_text': output_answers,
                # revert the key value pair
                'node_dict': {v: k for k, v in ele['node_data']['node_dict'].items()}
            }
            
            # Combine preprocessed data
            tokenized_data.append({
                "text_view": {
                    'ID': ID,
                    'question': question,
                    'input_ids': torch.LongTensor(tokenized_input["input_ids"]),
                    'attention_mask': torch.LongTensor(tokenized_input["attention_mask"]),
                    'answers': output_answers,
                    'entity_spans': entity_spans_sorted,
                    'entity_labels': entity_labels_sorted,
                },
                "graph_view": graph_view
            })
        return tokenized_data

    def __getitem__(self, idx):
        return self.dataset[idx]

    def preprocess_dataset(self, raw_dataset):
        """
        Generate `text_in`, `struct_in`, and `seq_out`.
        """
        preprocessed_data = []
        for ele in raw_dataset:
            if not ele["answers"] or not ele["answers"][0][1]:
                continue
            question, serialized_kg = self.kgqa_get_input(ele["question"], ele["kg_tuples"], ele["entities"])
            ele.update({"struct_in": serialized_kg, "text_in": question})
            preprocessed_data.append(ele)
        return preprocessed_data

    def kgqa_get_input(self, question, kg_tuples, entities):
        serialized_kg = self.serialize_kg_tuples(kg_tuples)
        serialized_entity = " ".join([": ".join(elm) for elm in entities]).strip()
        return question.strip(), f"{serialized_entity} | {serialized_kg}"

    def serialize_kg_tuples(self, kg_tuples):
        return " | ".join([" ".join(t) for t in kg_tuples])

    def get_entities_from_struct_in(self, struct_in):
        parts = struct_in.split('|')[1:]  # Skip the first part which is the serialized entities
        kg_entities = []
        for part in parts:
            elements = part.strip().split()
            if len(elements) >= 3:
                kg_entities.append(elements[0])
                kg_entities.append(elements[-1])
        return kg_entities

    def get_entity_token_indices(self, input_ids, entities, tokenizer):
        """
        Map entity spans in the tokenized sequence.
        """
        entity2tokens = {}
        input_token_ids = list(input_ids)
        
        unique_entities = list(set(entities))
        for entity in unique_entities:
            entity_tokens = tokenizer.tokenize(entity)
            entity_token_ids = tokenizer.convert_tokens_to_ids(entity_tokens)

            entity_spans = []
            for i in range(len(input_token_ids) - len(entity_token_ids) + 1):
                if input_token_ids[i:i + len(entity_token_ids)] == entity_token_ids:
                    entity_spans.append([i, i + len(entity_token_ids)-1])
            if entity_spans:
                # Only keep spans if they are found in the tokenized input
                entity2tokens[entity] = entity_spans
        return entity2tokens

def create_unified_mini_batch(samples):
    """
    Collate function for batching samples.
    """
    batch = {}
    
    text_input_ids = [s['text_view']['input_ids'] for s in samples]
    text_attention_mask = [s['text_view']['attention_mask'] for s in samples]
    entity_spans = [s['text_view']['entity_spans'] for s in samples]
    entity_labels = [s['text_view']['entity_labels'] for s in samples]
    
    batch['text_view'] = {
        'ID': [s['text_view']['ID'] for s in samples],
        'question': [s['text_view']['question'] for s in samples],
        'input_ids': pad_sequence(text_input_ids, batch_first=True),
        'attention_mask': pad_sequence(text_attention_mask, batch_first=True),
        'answers': [s['text_view']['answers'] for s in samples],
        'entity_spans': entity_spans,
        'entity_labels': entity_labels,
    }

    graph_input_ids = [s['graph_view']['input_ids'] for s in samples]
    graph_attention_mask = [s['graph_view']['attention_mask'] for s in samples]
    gnn_data = [s['graph_view']['gnn_data'] for s in samples]
    # graph_labels = [s['graph_view']['answers'] for s in samples]

    batch['graph_view'] = {
        'input_ids': pad_sequence(graph_input_ids, batch_first=True),
        'attention_mask': pad_sequence(graph_attention_mask, batch_first=True),
        'gnn_data': Batch.from_data_list(gnn_data),
        # 'labels': pad_sequence(graph_labels, batch_first=True, padding_value=-100),
        'answers': [s['graph_view']['answers_text'] for s in samples],
        'node_dict': [s['graph_view']['node_dict'] for s in samples],
    }
    return batch

def get_unified_dataloader(dataset, params, batch_size=16, shuffle=True, tokenizer=None):
    """
    Creates a DataLoader for the dataset.
    """
    unified_dataset = UnifiedDataset(dataset, params, tokenizer)
    return DataLoader(
        unified_dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=create_unified_mini_batch, num_workers=2, pin_memory=True
    )
