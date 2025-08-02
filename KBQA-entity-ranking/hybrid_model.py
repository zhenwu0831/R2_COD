import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridRankerModel(nn.Module):
    def __init__(self, params, text_model, graph_model):
        super(HybridRankerModel, self).__init__()
        self.text_model = text_model
        self.graph_model = graph_model
        self.params = params
        # self.layer_norm = nn.LayerNorm(self.params.hidden_dim)
        self.dropout = nn.Dropout(self.params.dropout)
        self.classifier = nn.Linear(params.hidden_dim, 1)
        self.ans_sigmoid = nn.Sigmoid()

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

    def forward(self, batch):
        device = next(self.parameters()).device

        graph_batch = batch["graph_view"]
        text_batch = batch["text_view"]

        # get text input output
        text_input_ids = text_batch['input_ids'].to(device)
        text_attention_mask = text_batch['attention_mask'].to(device)
        entity_spans = text_batch["entity_spans"]
        entity_labels = text_batch["entity_labels"]
        entity_embs = self.text_model(text_input_ids, text_attention_mask, entity_spans) 

        max_num_entities = entity_embs.shape[1]

        # Pad true labels for each batch item
        entity_labels_padded = []
        entity_mask = []

        for batch_idx, entity_list in enumerate(entity_spans):
            real_labels = [entity_labels[batch_idx].get(entity, 0) for entity in entity_list]
            padding_size = max_num_entities - len(entity_list)
            
            # Labels padded with 0s (same as before)
            labels_tensor = torch.tensor(
                real_labels + [0] * padding_size,  # Pad with 0s
                dtype=torch.float, device=device
            )

            # Mask padded with 0s (0 for padding, 1 for real entities)
            mask_tensor = torch.tensor(
                [1] * len(real_labels) + [0] * padding_size,  # 1 for real entities, 0 for padding
                dtype=torch.float, device=device
            )
            entity_labels_padded.append(labels_tensor)
            entity_mask.append(mask_tensor)

        # Convert to tensor (batch_size, max_entities)
        entity_labels_padded = torch.stack(entity_labels_padded)
        entity_mask = torch.stack(entity_mask)

        # get graph input output
        graph_outputs = self.graph_model(graph_batch)
        graph_encoder_hidden_states = graph_outputs['graph_embs'].to(device)
        head_nodes = graph_outputs['head_nodes']

        # import pdb; pdb.set_trace()

        masked_graph_embs = graph_encoder_hidden_states * (1 - head_nodes)

        # fuse the embeddings from the two modalities
        combined_embs = self.dropout(entity_embs + masked_graph_embs)
        final_output = self.ans_sigmoid(self.classifier(combined_embs).squeeze(-1))
        
        # pass through the two mlp heads
        gcn_output = self.gcn_head(masked_graph_embs)
        bert_output = self.bert_head(entity_embs)

        return final_output, gcn_output, bert_output, entity_labels_padded, entity_mask
