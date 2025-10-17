# components/event_embedder.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class PretrainedEventEmbedder(nn.Module):
    """
    Processes events where categorical features are already pre-embedded vectors.
    """
    def __init__(self, embedding_dim: int, num_feat_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        total_input_dim = embedding_dim + num_feat_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(total_input_dim),
            nn.Linear(total_input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, events_df: pd.DataFrame):
        device = next(self.parameters()).device
        act_emb = torch.from_numpy(np.stack(events_df['activity_embedding'].values)).float().to(device)
        res_emb = torch.from_numpy(np.stack(events_df['resource_embedding'].values)).float().to(device)
        semantic_emb = act_emb + res_emb # Combine via addition

        num_arr = events_df[['cost', 'time_from_start', 'time_from_previous']].values
        num_feats = torch.log1p(torch.as_tensor(num_arr, dtype=torch.float32, device=device).clamp_min(0))

        combined_input = torch.cat([semantic_emb, num_feats], dim=-1)
        return self.dropout(self.projection(combined_input))

class LearnedEventEmbedder(nn.Module):
    """
    Processes events by looking up learnable embeddings for categorical features (IDs).
    """
    def __init__(self, vocab_sizes: dict, embedding_dims: dict, num_feat_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.activity_embedding = nn.Embedding(
            vocab_sizes['activity'], embedding_dims['activity'], padding_idx=0
        )
        self.resource_embedding = nn.Embedding(
            vocab_sizes['resource'], embedding_dims['resource'], padding_idx=0
        )
        total_input_dim = embedding_dims['activity'] + embedding_dims['resource'] + num_feat_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(total_input_dim),
            nn.Linear(total_input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, events_df: pd.DataFrame):
        device = self.activity_embedding.weight.device
        act_ids = torch.from_numpy(events_df['activity_id'].values).long().to(device)
        res_ids = torch.from_numpy(events_df['resource_id'].values).long().to(device)
        act_emb = self.activity_embedding(act_ids)
        res_emb = self.resource_embedding(res_ids)

        num_arr = events_df[['cost', 'time_from_start', 'time_from_previous']].values
        num_feats = torch.log1p(torch.as_tensor(num_arr, dtype=torch.float32, device=device).clamp_min(0))

        combined_input = torch.cat([act_emb, res_emb, num_feats], dim=-1)
        return self.dropout(self.projection(combined_input))
