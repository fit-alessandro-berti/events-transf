# components/event_embedder.py
import torch
import torch.nn as nn
import pandas as pd


class EventEmbedder(nn.Module):
    """
    Processes event features, using learnable embeddings for categorical
    features (activity, resource) and combining them with numerical features
    before projecting to a single vector of size d_model.
    """
    def __init__(self, vocab_sizes: dict, embedding_dims: dict, num_feat_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # --- Embedding Layers for Categorical Features ---
        self.activity_embedding = nn.Embedding(
            vocab_sizes['activity'],
            embedding_dims['activity'],
            padding_idx=0
        )
        self.resource_embedding = nn.Embedding(
            vocab_sizes['resource'],
            embedding_dims['resource'],
            padding_idx=0
        )

        # Total input dimension after concatenation
        total_input_dim = embedding_dims['activity'] + embedding_dims['resource'] + num_feat_dim

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
            events_df (pd.DataFrame): DataFrame containing event features for a batch.
                                      Must contain 'activity_id', 'resource_id',
                                      and numerical feature columns.

        Returns:
            torch.Tensor: A tensor of shape (seq_len, d_model).
        """
        device = self.activity_embedding.weight.device

        # 1. Extract IDs and look up embeddings
        act_ids = torch.from_numpy(events_df['activity_id'].values).long().to(device)
        res_ids = torch.from_numpy(events_df['resource_id'].values).long().to(device)

        act_emb = self.activity_embedding(act_ids)  # Shape: (seq_len, activity_embedding_dim)
        res_emb = self.resource_embedding(res_ids)  # Shape: (seq_len, resource_embedding_dim)

        # 2. Extract and process numerical features
        num_arr = events_df[['cost', 'time_from_start', 'time_from_previous']].values
        num_feats = torch.as_tensor(num_arr, dtype=torch.float32, device=device)
        num_feats = torch.log1p(num_feats.clamp_min(0))  # Shape: (seq_len, num_feat_dim)

        # 3. Concatenate all features
        # Concatenating is generally better than adding for features from different sources
        combined_input = torch.cat([act_emb, res_emb, num_feats], dim=-1)

        # 4. Project to d_model
        final_emb = self.projection(combined_input)

        return self.dropout(final_emb)
