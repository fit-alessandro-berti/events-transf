# components/learned_event_embedder.py
import torch
import torch.nn as nn
import pandas as pd

class LearnedEventEmbedder(nn.Module):
    """
    Processes events by looking up learnable embeddings for categorical features (IDs),
    combining them with numerical features, and projecting to d_model.
    """
    def __init__(self, vocab_sizes: dict, embedding_dims: dict, num_feat_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        # --- Learnable Embedding Layers ---
        self.activity_embedding = nn.Embedding(
            vocab_sizes['activity'], embedding_dims['activity'], padding_idx=0
        )
        self.resource_embedding = nn.Embedding(
            vocab_sizes['resource'], embedding_dims['resource'], padding_idx=0
        )

        # Total input dimension is the sum of all feature dimensions
        total_input_dim = embedding_dims['activity'] + embedding_dims['resource'] + num_feat_dim

        # Projection layer to map the concatenated features to d_model
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
            events_df (pd.DataFrame): DataFrame with activity/resource IDs.

        Returns:
            torch.Tensor: A tensor of shape (seq_len, d_model).
        """
        device = self.activity_embedding.weight.device

        # 1. Extract IDs and look up learnable embeddings
        act_ids = torch.from_numpy(events_df['activity_id'].values).long().to(device)
        res_ids = torch.from_numpy(events_df['resource_id'].values).long().to(device)
        act_emb = self.activity_embedding(act_ids)
        res_emb = self.resource_embedding(res_ids)

        # 2. Extract and process numerical features
        num_arr = events_df[['cost', 'time_from_start', 'time_from_previous']].values
        num_feats = torch.log1p(torch.as_tensor(num_arr, dtype=torch.float32, device=device).clamp_min(0))

        # 3. Concatenate all features and project to d_model
        combined_input = torch.cat([act_emb, res_emb, num_feats], dim=-1)
        return self.dropout(self.projection(combined_input))
