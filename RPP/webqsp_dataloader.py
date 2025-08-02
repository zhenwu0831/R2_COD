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
    """
    Unified Dataset for text, graph, and combined views.
    """
    def __init__(self, dataset, params, global_label_map, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = self.preprocess_dataset(dataset)
        self.params = params
        self.label_map = global_label_map  # Use the global mapping

        self.dataset = self.tokenize_dataset()

    def __len__(self):
        return len(self.dataset)

    def map_entities(self, struct_in, node_dict):
        # Map entities in struct_in to their corresponding node indices in node_dict, e.g., "m.123" -> <E0>
        for entity, index in node_dict.items():
            struct_in = struct_in.replace(entity, f"<E{index}>")
        return struct_in

    def tokenize_dataset(self):
        """
        Pre-tokenize the dataset for all modes during initialization.
        """
        tokenized_data = []
        for ele in tqdm(self.dataset, desc="Tokenizing dataset"):
            output_answers = ele["answers"]
            iso = ele["isomorphism"]
            try:
                ID = ele["ID"]
            except:
                ID = ele["id"]
            question = ele["question"]

            label_idx = self.label_map[iso]

            struct_in = ele["struct_in"]
            # struct_in = self.map_entities(struct_in, ele['node_data']['node_dict'])

            # Text view
            seq_in = "{} ; structured knowledge: {}".format(ele["text_in"], struct_in)
            tokenized_input = self.tokenizer(
                seq_in, padding="max_length", truncation=True, max_length=self.params.input_max_length
            )
            s_expression = ele["seq_out"]
            tokenized_sexp = self.tokenizer(
                s_expression, padding="max_length", truncation=True, max_length=self.params.generation_max_length
            )
            tokenized_sexp_input_ids = torch.LongTensor(tokenized_sexp["input_ids"])
            tokenized_sexp_input_ids[tokenized_sexp_input_ids == self.tokenizer.pad_token_id] = -100

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

            # import pdb; pdb.set_trace()

            # Combine preprocessed data
            tokenized_data.append({
                "text_view": {
                    'ID': ID,
                    'question': question,
                    'input_ids': torch.LongTensor(tokenized_input["input_ids"]),
                    'attention_mask': torch.LongTensor(tokenized_input["attention_mask"]),
                    'labels': tokenized_sexp_input_ids,
                    'iso_labels': torch.tensor(label_idx, dtype=torch.long),
                    'answers': output_answers,
                    'gold_sexp': s_expression,
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
        # import pdb; pdb.set_trace() 
        preprocessed_data = []
        for ele in raw_dataset:
            if not ele["answers"] or not ele["answers"][0][1] or not ele['s_expression']:
                continue
            question, serialized_kg = self.kgqa_get_input(ele["question"], ele["kg_tuples"], ele["entities"])
            seq_out = ele["s_expression"]
            ele.update({"struct_in": serialized_kg, "text_in": question, "seq_out": seq_out})
            preprocessed_data.append(ele)
        return preprocessed_data

    def kgqa_get_input(self, question, kg_tuples, entities):
        serialized_kg = self.serialize_kg_tuples(kg_tuples)
        # serialized_entity = " ".join([elm[0] for elm in entities])
        serialized_entity = " ".join([": ".join(elm) for elm in entities]).strip()
        return question.strip(), f"{serialized_entity} | {serialized_kg}"

    def serialize_kg_tuples(self, kg_tuples):
        return " | ".join([" ".join(t) for t in kg_tuples])
    
    def id_2_answer_dict(self, dir):
        id2answer = {}
        for file in os.listdir(dir):
            with open(os.path.join(dir, file)) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    fid, answer = parts[0], parts[2] if len(parts) > 2 else "null"
                    id2answer[fid] = answer
        return id2answer

    def id_2_answer(self, freebase_id):
        return self.id2answer.get(freebase_id, "null")

def create_unified_mini_batch(samples):
    """
    Collate function for batching text, graph, or combined views.
    """
    batch = {}

    # Text view
    text_input_ids = [s['text_view']['input_ids'] for s in samples]
    text_attention_mask = [s['text_view']['attention_mask'] for s in samples]
    sexp_labels = [s['text_view']['labels'] for s in samples]

    batch['text_view'] = {
        'ID': [s['text_view']['ID'] for s in samples],
        'question': [s['text_view']['question'] for s in samples],
        'input_ids': pad_sequence(text_input_ids, batch_first=True),
        'attention_mask': pad_sequence(text_attention_mask, batch_first=True),
        'labels': pad_sequence(sexp_labels, batch_first=True, padding_value=-100),
        'answers': [s['text_view']['answers'] for s in samples],
        'gold_sexp': [s['text_view']['gold_sexp'] for s in samples],
        'iso_labels': torch.tensor([s['text_view']['iso_labels'] for s in samples], dtype=torch.long)
    }

    # Graph view
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


def get_unified_dataloader(dataset, params, batch_size=16, shuffle=True, global_label_map=None, tokenizer=None):
    """
    Creates a DataLoader for UnifiedDataset.
    """
    unified_dataset = UnifiedDataset(dataset, params, global_label_map, tokenizer)
    return DataLoader(
        unified_dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda samples: create_unified_mini_batch(samples),
        num_workers=2, pin_memory=True
    )