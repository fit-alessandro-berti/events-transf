# components.py

import torch
import torch.nn as nn
import math
from data_generator import SPECIAL_TOKENS


class EventEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model, cat_cardinalities, num_num_features, num_time_features):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cat_embs = nn.ModuleList([
            nn.Embedding(num_categories, d_model) for num_categories in cat_cardinalities
        ])
        self.num_mlp = nn.Sequential(
            nn.Linear(num_num_features, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(num_time_features, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.type_embed = nn.Embedding(2, d_model)

    def forward(self, token_ids, cat_feats, num_feats, time_feats):
        base_emb = self.token_embed(token_ids)
        event_mask = (token_ids == SPECIAL_TOKENS['<EVENT>']).unsqueeze(-1)

        cat_emb_sum = torch.zeros_like(base_emb)
        if len(self.cat_embs) > 0:
            for i, emb_layer in enumerate(self.cat_embs):
                cat_emb_sum += emb_layer(cat_feats[:, :, i])

        num_emb = self.num_mlp(num_feats)
        time_emb = self.time_mlp(time_feats)
        event_feature_emb = cat_emb_sum + num_emb + time_emb

        final_emb = torch.where(event_mask, event_feature_emb, base_emb)
        type_ids = (token_ids == SPECIAL_TOKENS['<EVENT>']).long()
        final_emb += self.type_embed(type_ids)

        return final_emb * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


# FIXED: Replaced TransformerDecoder with TransformerEncoder
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Use TransformerEncoderLayer, which is the standard self-attention block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, src_emb, attn_mask, padding_mask):
        src_with_pos = self.pos_encoder(src_emb)

        # The forward pass is now much cleaner
        output = self.transformer_encoder(
            src=src_with_pos,
            mask=attn_mask,  # This is our causal mask
            src_key_padding_mask=padding_mask
        )
        return output