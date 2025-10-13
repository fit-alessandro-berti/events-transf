# model.py

import torch
import torch.nn as nn
from components import EventEmbedder, DecoderOnlyTransformer
from data_generator import VOCAB_SIZE, ACTIVITY_VOCAB_SIZE, TIME_BUCKET_VOCAB_SIZE


class IOTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, cat_cardinalities, num_num_features, num_time_features, dropout=0.1):
        super().__init__()

        self.embedder = EventEmbedder(
            vocab_size=VOCAB_SIZE, d_model=d_model, cat_cardinalities=cat_cardinalities,
            num_num_features=num_num_features, num_time_features=num_time_features, dropout=dropout
        )

        self.backbone = DecoderOnlyTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout
        )

        self.next_activity_head = nn.Linear(d_model, ACTIVITY_VOCAB_SIZE)
        self.remaining_time_head = nn.Linear(d_model, TIME_BUCKET_VOCAB_SIZE)

    @staticmethod
    def _generate_causal_mask(sz: int, device: torch.device):
        """
        Return a boolean causal mask where True indicates positions that should be masked.
        """
        # upper triangular (excluding diagonal) is masked
        return torch.ones(sz, sz, dtype=torch.bool, device=device).triu(1)

    def forward(self, batch):
        token_ids = batch['tokens']
        cat_feats = batch['cat_feats']
        num_feats = batch['num_feats']
        time_feats = batch['time_feats']
        padding_mask = (batch['attention_mask'] == 0)  # [B,T] bool

        device = token_ids.device
        seq_len = token_ids.size(1)
        causal_mask = self._generate_causal_mask(seq_len, device)  # [T,T] bool

        x_emb = self.embedder(token_ids, cat_feats, num_feats, time_feats)
        h = self.backbone(x_emb, attn_mask=causal_mask, padding_mask=padding_mask)

        activity_logits = self.next_activity_head(h)    # [B,T,|A|]
        time_logits = self.remaining_time_head(h)       # [B,T,|B|]

        return activity_logits, time_logits
