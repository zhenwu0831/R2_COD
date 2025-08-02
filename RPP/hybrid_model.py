import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class HybridModel(nn.Module):
    def __init__(self, params, text_model, graph_model):
        super().__init__()
        self.params = params
        self.text_model = text_model
        self.graph_model = graph_model
        self.layer_norm = nn.LayerNorm(self.params.hidden_dim)
        self.dropout = nn.Dropout(self.params.dropout)
        self.classifier = nn.Linear(params.hidden_dim * 2, self.params.num_classes)
        # self.classifier = nn.Linear(2048 * 2, self.params.num_classes)
        
        self.gcn_head = nn.Sequential(
            nn.Linear(self.params.hidden_dim, 2048, bias=False),
            nn.Linear(2048, 2048, bias=False),
            nn.ReLU()
        ).cuda()
        
        self.bert_head = nn.Sequential(
            nn.Linear(self.params.hidden_dim, 2048, bias=False),
            nn.Linear(2048, 2048, bias=False),
            nn.ReLU()
        ).cuda()
    
    def forward(self, bat):
        device = next(self.parameters()).device
        
        # bert model forward
        text_batch = bat['text_view']
        text_input_ids = text_batch['input_ids'].to(device)
        text_attention_mask = text_batch['attention_mask'].to(device)
        _, bert_embs = self.text_model(text_input_ids, text_attention_mask)

        # graph model forward
        graph_batch = bat['graph_view']
        graph_outputs = self.graph_model(graph_batch)
        _, graph_embs = graph_outputs

        final_output = self.classifier(self.dropout(torch.cat([graph_embs, bert_embs], dim=-1)))
        # final_output = self.classifier(self.dropout(torch.cat([z_gcn_reps, z_bert_reps], dim=-1)))

        z_gcn_reps = self.gcn_head(graph_embs)
        z_bert_reps = self.bert_head(bert_embs)

        return final_output, z_gcn_reps, z_bert_reps
