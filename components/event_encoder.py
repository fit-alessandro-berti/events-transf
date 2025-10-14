# components/event_encoder.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input."""

    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # FIX: Initialize pe with a batch dimension for broadcasting
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Add positional encoding
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EventEncoder(nn.Module):
    """
    Encodes a sequence of event embeddings into a single context vector.
    Uses a Transformer Encoder.
    """

    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # FIX: Set batch_first=True to handle (batch, seq, feature) inputs and remove warning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src (torch.Tensor): Shape (batch_size, seq_len, d_model)
            src_key_padding_mask (torch.Tensor, optional): Shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Encoded representation, shape (batch_size, d_model)
        """
        # PositionalEncoding expects (seq_len, batch, d_model), so we permute
        src_permuted = src.permute(1, 0, 2)
        src_permuted = src_permuted * math.sqrt(self.d_model)
        src_permuted = self.pos_encoder(src_permuted)

        # Permute back to (batch_size, seq_len, d_model) for the transformer
        src_for_transformer = src_permuted.permute(1, 0, 2)

        output = self.transformer_encoder(src_for_transformer, src_key_padding_mask=src_key_padding_mask)

        # FIX: Use the representation of the last *actual* event, not the last padded token.
        if src_key_padding_mask is not None:
            # Get the length of each sequence in the batch
            lengths = (~src_key_padding_mask).sum(dim=1)
            # Create indices for batch and sequence dimensions
            batch_indices = torch.arange(output.size(0), device=output.device)
            # Select the hidden state at the last time step for each sequence
            encoded = output[batch_indices, lengths - 1]
        else:
            # If no mask, all sequences have full length
            encoded = output[:, -1, :]

        return encoded
