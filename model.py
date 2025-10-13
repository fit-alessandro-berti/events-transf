# model.py

import torch
import torch.nn as nn
from components import EventEmbedder, DecoderOnlyTransformer
from data_generator import VOCAB_SIZE, ACTIVITY_VOCAB_SIZE, TIME_BUCKET_VOCAB_SIZE, SPECIAL_TOKENS


class IOTransformer(nn.Module):
    """
    IO-Transformer for in-context learning on event data.
    """

    def __init__(self, d_model, n_layers, n_heads, num_num_features, num_time_features, dropout=0.1):
        super().__init__()

        self.embedder = EventEmbedder(VOCAB_SIZE, d_model, num_num_features, num_time_features)

        self.backbone = DecoderOnlyTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout
        )

        # Heads for different tasks
        self.next_activity_head = nn.Linear(d_model, ACTIVITY_VOCAB_SIZE)
        self.remaining_time_head = nn.Linear(d_model, TIME_BUCKET_VOCAB_SIZE)

    def _generate_causal_mask(self, sz, device):
        """Generates a square causal mask."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, batch):
        token_ids = batch['tokens']
        padding_mask = (batch['attention_mask'] == 0)  # True for positions to ignore

        device = token_ids.device
        seq_len = token_ids.size(1)

        # 1. Create causal attention mask
        causal_mask = self._generate_causal_mask(seq_len, device)

        # 2. Embed tokens
        x_emb = self.embedder(token_ids)

        # 3. Pass through Transformer backbone
        h = self.backbone(x_emb, tgt_mask=causal_mask, padding_mask=padding_mask)

        # 4. Apply task-specific heads
        # Note: We compute both for simplicity, loss is calculated on the relevant one.
        # This can be optimized by checking the task first.

        # The output vocabulary for activities starts after special tokens
        activity_logits = self.next_activity_head(h)

        # The output vocabulary for time buckets starts after activities
        time_logits = self.remaining_time_head(h)

        return activity_logits, time_logits
