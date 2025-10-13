# components.py

import torch
import torch.nn as nn
import math


# NEW: A proper event embedder as per the blueprint
class EventEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model, cat_cardinalities, num_num_features, num_time_features):
        super().__init__()
        self.d_model = d_model

        # Embedding for special tokens, labels, etc.
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Embeddings for categorical features
        # We'll embed each categorical feature and sum them up
        self.cat_embs = nn.ModuleList([
            nn.Embedding(num_categories, d_model) for num_categories in cat_cardinalities
        ])

        # MLPs for continuous features
        self.num_mlp = nn.Sequential(
            nn.Linear(num_num_features, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(num_time_features, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        # Layer to indicate token type (event vs. special)
        self.type_embed = nn.Embedding(2, d_model)  # 0 for special, 1 for event

    def forward(self, token_ids, cat_feats, num_feats, time_feats):
        # token_ids: [B, T], cat_feats: [B, T, n_cat], num_feats: [B, T, n_num], time_feats: [B, T, n_time]

        # Start with base embeddings for all tokens (special, labels, etc.)
        base_emb = self.token_embed(token_ids)

        # Create a mask for event tokens
        event_mask = (token_ids == SPECIAL_TOKENS['<EVENT>']).unsqueeze(-1)  # [B, T, 1]

        # 1. Categorical Embeddings
        cat_emb_sum = torch.zeros_like(base_emb)
        if len(self.cat_embs) > 0:
            for i, emb_layer in enumerate(self.cat_embs):
                cat_emb_sum += emb_layer(cat_feats[:, :, i])

        # 2. Numerical Embeddings
        num_emb = self.num_mlp(num_feats)

        # 3. Time Embeddings
        time_emb = self.time_mlp(time_feats)

        # The final embedding for an event is the sum of its feature embeddings
        event_feature_emb = cat_emb_sum + num_emb + time_emb

        # Use base_emb for special tokens and event_feature_emb for event tokens
        final_emb = torch.where(event_mask, event_feature_emb, base_emb)

        # Add a token type embedding
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
        # x shape: [B, T, E]
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dropout=0.1):
        super().__init__()
        # FIXED: Positional encoder is now correctly used
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def forward(self, src_emb, tgt_mask, padding_mask):
        # FIXED: Apply positional encoding
        src_with_pos = self.pos_encoder(src_emb)

        output = self.transformer_decoder(
            tgt=src_with_pos, memory=src_with_pos,  # Self-attention
            tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask
        )
        return output
