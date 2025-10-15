import torch
import torch.nn as nn


class EventEmbedder(nn.Module):
    """Embeds categorical and numerical event features into a single vector."""

    def __init__(self, cat_vocabs, num_feat_dim, d_model, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Embedding layers for categorical features
        self.act_embed = nn.Embedding(cat_vocabs['activity'], d_model // 2)
        self.res_embed = nn.Embedding(cat_vocabs['resource'], d_model // 2)

        # --- FIX: Normalize numeric features before the MLP (stabilizes training) ---
        self.num_norm = nn.LayerNorm(num_feat_dim)

        # MLP for numerical features (cost, time_from_start, time_from_previous)
        self.num_mlp = nn.Sequential(
            nn.Linear(num_feat_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

        # Projection layer to combine embeddings + a normalization for stability
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, events_df):
        """
        Args:
            events_df (pd.DataFrame): DataFrame containing event features for a batch.

        Returns:
            torch.Tensor: A tensor of shape (seq_len, d_model).
        """
        device = self.act_embed.weight.device

        # Categorical features
        activities = torch.as_tensor(
            events_df['activity'].values, dtype=torch.long, device=device
        )
        resources = torch.as_tensor(
            events_df['resource'].values, dtype=torch.long, device=device
        )

        act_emb = self.act_embed(activities)
        res_emb = self.res_embed(resources)
        cat_emb = torch.cat([act_emb, res_emb], dim=-1)  # (seq_len, d_model)

        # Numerical features
        num_arr = events_df[['cost', 'time_from_start', 'time_from_previous']].values
        num_feats = torch.as_tensor(num_arr, dtype=torch.float32, device=device)

        # --- FIX: Apply normalization to raw numerical features ---
        num_feats = self.num_norm(num_feats)
        num_emb = self.num_mlp(num_feats)  # (seq_len, d_model)

        # Combine and project
        combined_emb = torch.cat([cat_emb, num_emb], dim=-1)  # (seq_len, 2*d_model)
        final_emb = self.proj(combined_emb)  # (seq_len, d_model)

        return self.dropout(final_emb)
