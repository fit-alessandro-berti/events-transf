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
    aggregates them with a light attention-pooling head (mask-aware).
    """

    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model

        # NEW: Use Pre-Norm (norm_first=True) for more stable optimization.
        # NEW: Keep the feed-forward width small and proportional to d_model.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=max(4 * d_model, d_model),  # 4x width keeps model small & stable
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # NEW: Attention pooling over time to produce a robust sequence representation.
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
        # Standard Transformer scaling before adding positions
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Encode
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        # output: (B, T, d_model)

        # Mask-aware attention pooling over time
        attn_logits = self.attn_pool(output).squeeze(-1)  # (B, T)
        if src_key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(src_key_padding_mask, float('-inf'))

        # Softmax across time; if a row had all pads (shouldn't happen), softmax would be NaN,
        # but upstream guarantees at least one valid token per sequence.
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T)
        encoded = (attn_weights.unsqueeze(-1) * output).sum(dim=1)  # (B, d_model)

        # Final normalization for stability
        encoded = self.out_norm(encoded)
        return encoded
