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
    Encodes a sequence of event embeddings using a Transformer Encoder.
    """

    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # FIX: Set batch_first=True to handle (batch, seq, feature) inputs directly.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src (torch.Tensor): Shape (batch_size, seq_len, d_model)
            src_key_padding_mask (torch.Tensor, optional): Shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Encoded representation of the last valid token for each sequence.
                          Shape (batch_size, d_model)
        """
        # Input is already (batch, seq, feature), no permutation needed.
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # FIX: Robustly get the representation of the last *actual* event, not a padded one.
        if src_key_padding_mask is not None:
            # Get the length of each sequence in the batch
            lengths = (~src_key_padding_mask).sum(dim=1)
            # Create indices for batch dimension
            batch_indices = torch.arange(output.size(0), device=output.device)
            # Clamp lengths to avoid index -1 for empty sequences (edge case)
            last_token_indices = torch.clamp(lengths - 1, min=0)
            # Select the hidden state at the last time step for each sequence
            encoded = output[batch_indices, last_token_indices]
        else:
            # If no mask, all sequences have full length
            encoded = output[:, -1, :]

        return encoded
