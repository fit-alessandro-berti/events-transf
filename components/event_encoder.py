# components/event_encoder.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input. Adapted for batch_first=True."""

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Use a batch dimension of 1 for easy broadcasting [1, max_len, d_model]
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding up to the sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EventEncoder(nn.Module):
    """
    Encodes a sequence of event embeddings using a Transformer Encoder and
    aggregates them with a hybrid CLS + mask-aware attention pooling head.
    """

    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Pre-Norm transformer (stable) with modest FF width
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=max(4 * d_model, d_model),
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Attention pooling over time (excluding CLS) to produce a robust sequence representation.
        self.attn_pool = nn.Linear(d_model, 1)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src (torch.Tensor): Shape (batch_size, seq_len, d_model)
            src_key_padding_mask (torch.Tensor, optional): Shape (batch_size, seq_len), True for pads.

        Returns:
            torch.Tensor: Aggregated representation for each sequence.
                          Shape (batch_size, d_model)
        """
        B, T, D = src.shape

        # Prepend CLS
        cls = self.cls_token.expand(B, 1, D)
        src = torch.cat([cls, src], dim=1)  # (B, 1+T, D)

        # Extend mask (CLS is never padded)
        if src_key_padding_mask is not None:
            pad_col = torch.zeros((B, 1), dtype=torch.bool, device=src_key_padding_mask.device)
            src_key_padding_mask = torch.cat([pad_col, src_key_padding_mask], dim=1)  # (B, 1+T)

        # Standard Transformer scaling before adding positions
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Encode
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)  # (B, 1+T, D)

        # Split CLS and tokens
        cls_out = output[:, 0, :]        # (B, D)
        tokens = output[:, 1:, :]        # (B, T, D)

        # Build mask for tokens only (exclude CLS)
        token_mask = None
        if src_key_padding_mask is not None:
            token_mask = src_key_padding_mask[:, 1:]  # (B, T)

        # Mask-aware attention pooling on tokens
        attn_logits = self.attn_pool(tokens).squeeze(-1)  # (B, T)
        if token_mask is not None:
            attn_logits = attn_logits.masked_fill(token_mask, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T)
        pooled = (attn_weights.unsqueeze(-1) * tokens).sum(dim=1)  # (B, D)

        # Hybrid aggregation: CLS + pooled
        encoded = 0.5 * (cls_out + pooled)
        encoded = self.out_norm(encoded)
        return encoded
