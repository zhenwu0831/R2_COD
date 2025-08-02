import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import T5EncoderModel, AutoTokenizer

class T5EntityRanker(nn.Module):
    def __init__(self, model_name="t5-base", tokenizer=None):
        super(T5EntityRanker, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.entity_scorer = nn.Linear(self.encoder.config.d_model, 1)  # Classifier for entity relevance
        self.ans_sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, entity_spans):
        """
        Encode input and extract entity representations.
        """
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        entity_embeddings = self.extract_entity_embeddings(encoder_outputs, entity_spans)
        entity_embeddings = [emb.to("cuda") for emb in entity_embeddings]
        entity_embeddings_padded = pad_sequence(entity_embeddings, batch_first=True)

        # Compute entity scores
        # entity_scores = self.entity_scorer(entity_embeddings_padded).squeeze(-1) 
        # entity_scores = self.ans_sigmoid(entity_scores)

        # import pdb; pdb.set_trace()  # Debugging line to check entity scores

        # return entity_scores  # Raw scores (apply sigmoid in loss function)
        return entity_embeddings_padded  # Return padded entity embeddings for further processing

    def extract_entity_embeddings(self, encoder_outputs, entity_spans):
        """
        Extract entity token embeddings by averaging the token embeddings in the entity span.
        """
        batch_size = encoder_outputs.shape[0]
        entity_embeddings = []
        batch_entity_counts = []

        for batch_idx in range(batch_size):
            batch_entity_embs = []
            for entity, spans in entity_spans[batch_idx].items():
                token_embs = []
                for start, end in spans:
                    token_embs.append(encoder_outputs[batch_idx, start:end + 1].mean(dim=0))  # Average across tokens

                if token_embs:
                    batch_entity_embs.append(torch.stack(token_embs).mean(dim=0))  # Average across spans

            if batch_entity_embs:
                entity_embeddings.append(torch.stack(batch_entity_embs))  # Collect all entity embeddings
            else:
                entity_embeddings.append(torch.zeros((1, encoder_outputs.shape[-1]), device=encoder_outputs.device))

            batch_entity_counts.append(len(batch_entity_embs))
        
        # print(f"Extracted entities per batch: {batch_entity_counts}")
        # import pdb; pdb.set_trace() 

        return entity_embeddings  # Shape: (num_entities, hidden_dim)