# components/event_embedder.py
import torch
import torch.nn as nn


class EventEmbedder(nn.Module):
    """
    Embeds categorical and numerical event features into a single vector.

    Targeted improvements:
    - Reserve 0 as PAD via padding_idx=0 to keep pad embeddings zero and stable.
    - Log1p-transform numeric features before LayerNorm (robust under heavy tails).
    - Lightweight FiLM-style gating where numeric stream modulates categorical.
    """

    def __init__(self, cat_vocabs, num_feat_dim, d_model, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Embedding layers for categorical features (0 is PAD)
        self.act_embed = nn.Embedding(cat_vocabs['activity'], d_model // 2, padding_idx=0)
        self.res_embed = nn.Embedding(cat_vocabs['resource'], d_model // 2, padding_idx=0)

        # Normalize numeric features before the MLP (stabilizes training)
        self.num_norm = nn.LayerNorm(num_feat_dim)

        # MLP for numerical features (cost, time_from_start, time_from_previous)
        self.num_mlp = nn.Sequential(
            nn.Linear(num_feat_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

        # FiLM-style gating: numeric → (gamma, beta) to modulate categorical branch
        self.film_gamma = nn.Linear(d_model, d_model)  # produces scale
        self.film_beta = nn.Linear(d_model, d_model)   # produces shift

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

        # Categorical features (0 is PAD, reserved)
        activities = torch.as_tensor(
            events_df['activity'].values, dtype=torch.long, device=device
        )
        resources = torch.as_tensor(
            events_df['resource'].values, dtype=torch.long, device=device
        )

        act_emb = self.act_embed(activities)
        res_emb = self.res_embed(resources)
        cat_emb = torch.cat([act_emb, res_emb], dim=-1)  # (seq_len, d_model)

        # Numerical features: log1p + LayerNorm + MLP
        num_arr = events_df[['cost', 'time_from_start', 'time_from_previous']].values
        num_feats = torch.as_tensor(num_arr, dtype=torch.float32, device=device)
        # robustify skew/heavy tails
        num_feats = torch.log1p(num_feats.clamp_min(0))
        num_feats = self.num_norm(num_feats)
        num_emb = self.num_mlp(num_feats)  # (seq_len, d_model)

        # FiLM gating: numeric modulates categorical
        gamma = torch.sigmoid(self.film_gamma(num_emb))
        beta = self.film_beta(num_emb)
        cat_mod = cat_emb * gamma + beta

        # Ensure padded rows stay zero after FiLM modulation
        is_pad = (activities == 0) & (resources == 0)
        if is_pad.any():
            cat_mod = cat_mod.masked_fill(is_pad.unsqueeze(-1), 0.0)
            num_emb = num_emb.masked_fill(is_pad.unsqueeze(-1), 0.0)

        # Combine and project
        combined_emb = torch.cat([cat_mod, num_emb], dim=-1)  # (seq_len, 2*d_model)
        final_emb = self.proj(combined_emb)  # (seq_len, d_model)

        return self.dropout(final_emb)
