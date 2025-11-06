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
    aggregates them with a hybrid CLS + mask-aware Multi-Head Attention (MHA)
    pooling head.

    Update:
    - Replaced simple attention pooling with a more powerful MHA-based pooling.
    - A learnable 'pool_query' vector attends to the output tokens.
    - Replaced the simple scalar mix with a concatenation and projection layer
      to non-linearly combine the CLS token and the pooled token representation.
      This provides a more expressive and powerful aggregation mechanism.
    """

    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Pre-Norm transformer (stable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,  # Standard FFN width
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- New Powerful Aggregation Head ---
        # 1. A learnable query vector for MHA pooling
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pool_query, std=0.02)

        # 2. A Multi-Head Attention layer to perform the pooling
        self.mha_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. A projection layer to combine CLS and Pooled representations
        self.final_projection = nn.Linear(d_model * 2, d_model)
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

        # --- MHA-based Pooling ---
        # Use the learnable pool_query to attend to the sequence tokens
        pool_q = self.pool_query.expand(B, -1, -1)  # (B, 1, D)
        pooled_out, _ = self.mha_pool(
            query=pool_q,
            key=tokens,
            value=tokens,
            key_padding_mask=token_mask
        )
        pooled = pooled_out.squeeze(1)  # (B, 1, D) -> (B, D)

        # --- Hybrid Aggregation ---
        # Concatenate and project for a powerful, flexible combination
        concatenated = torch.cat([cls_out, pooled], dim=-1)  # (B, 2*D)
        projected = self.final_projection(concatenated)      # (B, D)

        # Final normalization
        encoded = self.out_norm(projected)
        return encoded
