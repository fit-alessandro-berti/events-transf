# /io_transformer/model.py

import torch
import torch.nn as nn
from embedding import EventEmbedder


class IOTransformer(nn.Module):
    """
    Decoder-only Transformer for in-context learning on event data.
    """

    def __init__(self, vocab_sizes, d_model=128, n_head=4, n_layers=4, dropout=0.1, num_numeric_features=1):
        super().__init__()
        self.d_model = d_model
        self.embedder = EventEmbedder(vocab_sizes, d_model, num_numeric_features)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=False  # Our embedder outputs (S, N, E)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Prediction heads
        self.cls_head = nn.Linear(d_model, vocab_sizes['activity'])

        # Predict mu and log_sigma for a LogNormal distribution
        self.reg_head = nn.Linear(d_model, 2)

    def forward(self, batch):
        # Get embeddings and masks
        # src shape: (S, N, E)
        src = self.embedder(batch)

        # Masks shape: (N, S) -> need (S, S) for causal and (N, S) for padding
        padding_mask = batch['padding_mask']
        causal_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)

        # Pass through transformer decoder
        # Note: TransformerDecoder wants target, memory, tgt_mask, memory_mask, tgt_key_padding_mask
        # For decoder-only, target and memory are the same.
        output = self.transformer_decoder(
            tgt=src,
            memory=src,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask
        )

        # Heads
        # output shape: (S, N, E) -> permute to (N, S, E)
        output = output.permute(1, 0, 2)

        cls_logits = self.cls_head(output)
        reg_params = self.reg_head(output)

        mu, log_sigma = torch.chunk(reg_params, 2, dim=-1)

        return {
            'classification': cls_logits,
            'regression': (mu.squeeze(-1), log_sigma.squeeze(-1))
        }

    def generate_square_subsequent_mask(self, sz: int):
        """Generates a square causal mask for the sequence."""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
