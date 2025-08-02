import torch, torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5EncoderModel, T5Model, AutoModel
from torch_geometric.nn import FastRGCNConv, RGCNConv, RGATConv, GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import dropout_edge
import numpy as np
import json
import time

class GNNDropout(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.drop = nn.Dropout(params.dropout)
    
    def forward(self, inp):
        x, edge_index, edge_type = inp
        return self.drop(x), edge_index, edge_type


class GNNReLu(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.relu = nn.ReLU()
  
    def forward(self, inp):
        x, edge_index, edge_type = inp
        return self.relu(x), edge_index, edge_type


class GNNSoftmax(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.softmax = nn.Softmax()
  
    def forward(self, inp):
        x, edge_index, edge_type = inp
        return self.softmax(x), edge_index, edge_type



class GNNConv(torch.nn.Module):
    def __init__(self, params, inp_dim, out_dim):
        super().__init__()
        self.params = params
        
        if self.params.gnn_model == 'rgcn':
            self.gnn 	= FastRGCNConv(in_channels= inp_dim, out_channels=out_dim, num_relations=self.params.num_rels)
        elif self.params.gnn_model == 'rgat':
            self.gnn 	= RGATConv(in_channels= inp_dim, out_channels=out_dim, num_relations=self.params.num_rels)
        elif self.params.gnn_model == 'gcn':
            self.gnn    = GCNConv(in_channels= inp_dim, out_channels=out_dim)
  
    def forward(self, inp):
        x, edge_index, edge_type = inp
        return self.gnn(x, edge_index, edge_type), edge_index, edge_type


class DeepNet(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.gnn_layers = []

        self.gnn_layers.append(GNNConv(self.params, self.params.hidden_dim * 2, self.params.hidden_dim))
        self.gnn_layers.append(GNNSoftmax(self.params))
        self.gnn_layers.append(GNNDropout(self.params))

        for i in range(self.params.gnn_depth-2):
            self.gnn_layers.append(GNNConv(self.params, self.params.hidden_dim, self.params.hidden_dim))
            self.gnn_layers.append(GNNSoftmax(self.params))
            self.gnn_layers.append(GNNDropout(self.params))

        self.gnn_layers.append(GNNConv(self.params, self.params.hidden_dim, self.params.hidden_dim))
        self.gnn_layers.append(GNNSoftmax(self.params))
        self.gnn_layers.append(GNNDropout(self.params))

        self.gnn_module = nn.Sequential(*self.gnn_layers)
  
        
    def forward(self, x, edge_index, edge_type):

        x,edge_index, edge_type = self.gnn_module((x, edge_index, edge_type))
        
        return x

class GNNClassifier(nn.Module):
    def __init__(self, params, tokenizer):
        super().__init__()
        self.params = params
        # self.ptlm_model = T5Model.from_pretrained('t5-base')
        self.ptlm_model = AutoModel.from_pretrained(self.params.ptlm_model)
        self.ptlm_model.resize_token_embeddings(len(tokenizer))
        self.gnn = DeepNet(params)
        self.layer_norm = nn.LayerNorm(self.params.hidden_dim)
        self.pooling = global_max_pool  

        self.graph_classifier = nn.Linear(self.params.hidden_dim, self.params.num_classes)

    def forward(self, bat):
        device = next(self.parameters()).device
        input_ids, attention_mask, gnn_data = bat['input_ids'], bat['attention_mask'], bat['gnn_data']

        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        gnn_data.x = gnn_data.x.to(device)

        if self.params.ptlm_model == 'bert-base-uncased':
            ptlm_embs = self.ptlm_model(input_ids=input_ids, attention_mask=attention_mask)
            question_emb = ptlm_embs.last_hidden_state[:, 0, :]  # CLS token embedding
        elif self.params.ptlm_model == 't5-base':
            ptlm_embs = self.ptlm_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            question_emb = ptlm_embs.last_hidden_state.mean(dim=1)

        edge_index, edge_type = gnn_data.edge_index.to(device), gnn_data.edge_type.to(device)
        head_nodes = gnn_data.head_x
        gnn_batch = gnn_data.batch.to(device)

        question_emb_expanded = question_emb[gnn_batch]
        head_nodes_expanded = head_nodes.unsqueeze(-1).float().to(device)

        combined_features = torch.cat([question_emb_expanded, gnn_data.x], dim=-1)
        combined_features = combined_features.to(device)

        graph_embs = self.gnn(combined_features, edge_index, edge_type)
        graph_embs = self.layer_norm(graph_embs)

        graph_representation = self.pooling(graph_embs, gnn_batch)

        logits = self.graph_classifier(graph_representation)

        return logits, graph_representation
