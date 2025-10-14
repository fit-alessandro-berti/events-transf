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
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src (torch.Tensor): Shape (seq_len, batch_size, d_model)
            src_key_padding_mask (torch.Tensor, optional): Shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Encoded representation, shape (batch_size, d_model)
        """
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # Transformer expects (seq_len, batch, d_model)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        # Use the representation of the last token (mean pooling is also an option)
        encoded = output[-1, :, :]
        return encoded
