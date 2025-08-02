import torch, torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import FastRGCNConv, RGCNConv, RGATConv, GCNConv, GATConv
from torch_geometric.nn.models import GraphSAGE
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

        self.gnn_layers.append(GNNConv(self.params, self.params.hidden_dim + self.params.text_emb_dim + 1, self.params.hidden_dim))
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

class EnhancedGNNEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.ptlm_model = AutoModel.from_pretrained(self.params.ptlm_model)
        self.gnn = DeepNet(params)
        self.ans_classifier = nn.Linear(self.params.hidden_dim, 1)
        self.ans_sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(self.params.hidden_dim)
        self.eps = self.params.eps

    def forward(self, bat):
        device = next(self.parameters()).device
        input_ids, attention_mask, gnn_data = bat['input_ids'], bat['attention_mask'], bat['gnn_data']

        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        gnn_data.x = gnn_data.x.to(device)

        ptlm_embs = self.ptlm_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        question_emb = ptlm_embs['last_hidden_state'].mean(dim=1)

        x, edge_index, edge_type = gnn_data.x, gnn_data.edge_index, gnn_data.edge_type
        head_nodes, answer_nodes = gnn_data.head_x, gnn_data.answer_x
        gnn_batch = gnn_data.batch

        gnn_batch = gnn_data.batch.to(device)
        head_nodes = head_nodes.to(device)
        answer_nodes = answer_nodes.to(device)

        question_emb_expanded = question_emb[gnn_batch]
        head_nodes_expanded = head_nodes.unsqueeze(-1).float().to(device)

        combined_features = torch.cat([question_emb_expanded, x, head_nodes_expanded], dim=-1)
        combined_features, edge_index, edge_type = combined_features.to(device), edge_index.to(device), edge_type.to(device)

        graph_embs = self.gnn(combined_features, edge_index, edge_type)
        graph_embs = self.layer_norm(graph_embs)

        # updated_node_features = torch.cat([question_emb_expanded, graph_embs, head_nodes_expanded], dim=-1)

        updated_node_features = graph_embs

        max_nodes = torch.bincount(gnn_batch).max().item()
        batch_size = question_emb.shape[0]
        emb_dim = updated_node_features.size(-1)

        padded_embs = torch.zeros((batch_size, max_nodes, emb_dim), device=device)
        node_mask = torch.arange(max_nodes, device=device).unsqueeze(0) < torch.bincount(gnn_batch).unsqueeze(1)
        padded_embs[node_mask] = updated_node_features

        padded_head_nodes = torch.zeros((batch_size, max_nodes, 1), device=device)
        padded_head_nodes[node_mask] = head_nodes_expanded

        padded_answer_nodes = torch.zeros((batch_size, max_nodes, 1), device=device, dtype=torch.long)
        padded_answer_nodes[node_mask] = answer_nodes.unsqueeze(-1)

        return {
            'head_nodes': padded_head_nodes,
            'batch_idxs': gnn_batch,
            'answer_nodes': padded_answer_nodes,
            'graph_embs': padded_embs,
            'node_mask': node_mask,
        }

class CrossAttentionGNNEncoder(nn.Module):
    def __init__(self, params, use_concatenation=True):
        super().__init__()
        self.params = params
        self.ptlm_model = AutoModel.from_pretrained(self.params.ptlm_model)
        self.gnn = DeepNet(params)
        self.ans_classifier = nn.Linear(self.params.node_emb_dim, 1)
        self.ans_sigmoid = nn.Sigmoid()
        self.attention = nn.MultiheadAttention(embed_dim=self.params.hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.params.hidden_dim)
        self.eps = self.params.eps
        self.use_concatenation = use_concatenation
        # self.query_projection = nn.Linear(self.ptlm_model.config.hidden_size, self.params.node_emb_dim, bias=False)

    def forward(self, bat):
        device = next(self.parameters()).device
        input_ids, attention_mask, gnn_data = bat['input_ids'], bat['attention_mask'], bat['gnn_data']

        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        gnn_data.x = gnn_data.x.to(device)

        ptlm_embs = self.ptlm_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        question_emb = ptlm_embs['last_hidden_state']
        question_emb_pooled = question_emb.mean(dim=1)

        x, edge_index, edge_type = gnn_data.x, gnn_data.edge_index, gnn_data.edge_type
        head_nodes, answer_nodes = gnn_data.head_x, gnn_data.answer_x
        gnn_batch = gnn_data.batch.to(device)
        head_nodes = head_nodes.to(device)
        answer_nodes = answer_nodes.to(device)

        question_emb_expanded = question_emb_pooled[gnn_batch]
        head_nodes_expanded = head_nodes.unsqueeze(-1).float().to(device)

        combined_features = torch.cat([question_emb_expanded, x, head_nodes_expanded], dim=-1)
        combined_features = combined_features.to(device)
        graph_embs = self.gnn(combined_features, edge_index.to(device), edge_type.to(device))
        graph_embs = self.layer_norm(graph_embs)

        updated_node_features = (torch.cat([question_emb_expanded, graph_embs, head_nodes_expanded], dim=-1)
                                 if self.use_concatenation else graph_embs)

        max_nodes = torch.bincount(gnn_batch).max().item()
        batch_size = question_emb.size(0)
        emb_dim = updated_node_features.size(-1)

        padded_graph_embs = torch.zeros((batch_size, max_nodes, emb_dim), device=device)
        node_mask = torch.arange(max_nodes, device=device).unsqueeze(0) < torch.bincount(gnn_batch).unsqueeze(1)
        padded_graph_embs[node_mask] = updated_node_features

        padded_head_nodes = torch.zeros((batch_size, max_nodes, 1), device=device)
        padded_head_nodes[node_mask] = head_nodes_expanded

        padded_answer_nodes = torch.zeros((batch_size, max_nodes, 1), device=device, dtype=torch.long)
        padded_answer_nodes[node_mask] = answer_nodes.unsqueeze(-1)

        key_padding_mask = (attention_mask == 0).bool().to(device)

        # question_emb = self.query_projection(question_emb).to(device)
        attn_output, _ = self.attention(padded_graph_embs, question_emb, question_emb, key_padding_mask=key_padding_mask)
        node_embs_after_attention = attn_output * node_mask.unsqueeze(-1).float()

        return {
            'graph_embs': node_embs_after_attention,
            'head_nodes': padded_head_nodes,
            'answer_nodes': padded_answer_nodes,
            'batch_idxs': gnn_batch,
            'node_mask': node_mask,
        }
