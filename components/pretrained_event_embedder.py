# components/pretrained_event_embedder.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class PretrainedEventEmbedder(nn.Module):
    """
    Processes events where categorical features are already pre-embedded vectors.
    It combines these with numerical features and projects them to d_model.
    """
    def __init__(self, embedding_dim: int, num_feat_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Total input dimension is the pre-trained embedding dim + number of numerical features
        total_input_dim = embedding_dim + num_feat_dim

        # A single projection layer (MLP) to map the combined input to d_model
        self.projection = nn.Sequential(
            nn.LayerNorm(total_input_dim),
            nn.Linear(total_input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, events_df: pd.DataFrame):
        """
        Args:
            events_df (pd.DataFrame): DataFrame with pre-computed embeddings.

        Returns:
            torch.Tensor: A tensor of shape (seq_len, d_model).
        """
        device = next(self.parameters()).device

        # 1. Extract pre-computed semantic embeddings from the DataFrame
        act_emb = torch.from_numpy(np.stack(events_df['activity_embedding'].values)).float().to(device)
        res_emb = torch.from_numpy(np.stack(events_df['resource_embedding'].values)).float().to(device)
        # Combine embeddings via addition to merge their semantic information
        semantic_emb = act_emb + res_emb

        # 2. Extract and process numerical features
        num_arr = events_df[['cost', 'time_from_start', 'time_from_previous']].values
        num_feats = torch.log1p(torch.as_tensor(num_arr, dtype=torch.float32, device=device).clamp_min(0))

        # 3. Concatenate all features and project to d_model
        combined_input = torch.cat([semantic_emb, num_feats], dim=-1)
        return self.dropout(self.projection(combined_input))
