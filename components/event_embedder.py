# components/event_embedder.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class EventEmbedder(nn.Module):
    """
    Processes event features, where activity/resource are pre-embedded vectors,
    into a single vector of size d_model.
    """

    def __init__(self, embedding_dim, num_feat_dim, d_model, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Total input dimension is the semantic embedding dimension + number of numerical features
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
            events_df (pd.DataFrame): DataFrame containing event features for a batch.
                                      Must contain 'activity_embedding', 'resource_embedding',
                                      and numerical feature columns.

        Returns:
            torch.Tensor: A tensor of shape (seq_len, d_model).
        """
        device = next(self.parameters()).device

        # 1. Extract and combine semantic embeddings
        # The DataFrame contains numpy arrays, so we stack them into a single tensor.
        act_emb = torch.from_numpy(np.stack(events_df['activity_embedding'].values)).float().to(device)
        res_emb = torch.from_numpy(np.stack(events_df['resource_embedding'].values)).float().to(device)

        # Combine embeddings via addition to merge their semantic information
        semantic_emb = act_emb + res_emb  # Shape: (seq_len, embedding_dim)

        # 2. Extract and process numerical features
        num_arr = events_df[['cost', 'time_from_start', 'time_from_previous']].values
        num_feats = torch.as_tensor(num_arr, dtype=torch.float32, device=device)

        # Log1p transform for robustness against skewed distributions
        num_feats = torch.log1p(num_feats.clamp_min(0))  # Shape: (seq_len, num_feat_dim)

        # 3. Concatenate all features
        combined_input = torch.cat([semantic_emb, num_feats], dim=-1)  # Shape: (seq_len, embedding_dim + num_feat_dim)

        # 4. Project to d_model
        final_emb = self.projection(combined_input)

        # Ensure that padded rows result in a zero vector embedding
        # A simple way is to check if time_from_start is 0 and it's not the first event in a sequence.
        # However, the padding mask from the MetaLearner will handle this more robustly at the encoder stage.
        # Let's assume the padding mask takes care of it.

        return self.dropout(final_emb)
