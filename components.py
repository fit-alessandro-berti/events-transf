# components.py

import torch
import torch.nn as nn
import math


class EventEmbedder(nn.Module):
    """
    Embeds categorical, numerical, and time features of an event.
    This version is simplified to work with placeholder token IDs.
    """

    def __init__(self, vocab_size, d_model, num_num_features, num_time_features):
        super().__init__()
        # Generic embedding for all tokens (special, activity, time buckets)
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # MLPs for continuous features (would be used in a more advanced setup)
        # For this example, we rely on the token_embed for simplicity.
        self.num_mlp = nn.Sequential(
            nn.Linear(num_num_features, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, d_model)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(num_time_features, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, d_model)
        )
        self.d_model = d_model

    def forward(self, token_ids):
        # In this simplified implementation, we directly embed the token IDs.
        # A full implementation would combine token embeddings with MLP outputs
        # for continuous features associated with each event token.
        return self.token_embed(token_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Adds positional information to the input embeddings."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    A standard causal Transformer decoder backbone.
    """

    def __init__(self, d_model, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, src_emb, tgt_mask, padding_mask):
        # PyTorch's TransformerDecoder expects memory, which we don't have.
        # We use it here as a stack of self-attention layers.
        # It also expects tgt_mask for causality.

        # Reshape for positional encoding if batch_first
        # src_emb = src_emb.permute(1, 0, 2) # (T, B, E)
        # src_emb = self.pos_encoder(src_emb)
        # src_emb = src_emb.permute(1, 0, 2) # (B, T, E)

        # The decoder layer will apply the target mask for causality
        output = self.transformer_decoder(
            tgt=src_emb,
            memory=src_emb,  # Self-attention
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask
        )
        return output
