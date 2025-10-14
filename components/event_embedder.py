# components/event_embedder.py
import torch
import torch.nn as nn


class EventEmbedder(nn.Module):
    """Embeds categorical and numerical event features into a single vector."""

    def __init__(self, cat_vocabs, num_feat_dim, d_model):
        super().__init__()
        self.d_model = d_model

        # Embedding layers for categorical features
        self.act_embed = nn.Embedding(cat_vocabs['activity'], d_model // 2)
        self.res_embed = nn.Embedding(cat_vocabs['resource'], d_model // 2)

        # MLP for numerical features (cost, time_from_start, time_from_previous)
        self.num_mlp = nn.Sequential(
            nn.Linear(num_feat_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

        # Projection layer to combine embeddings
        self.proj = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, events_df):
        """
        Args:
            events_df (pd.DataFrame): DataFrame containing event features for a batch.

        Returns:
            torch.Tensor: A tensor of shape (seq_len, d_model).
        """
        # Categorical features
        activities = torch.LongTensor(events_df['activity'].values)
        resources = torch.LongTensor(events_df['resource'].values)

        act_emb = self.act_embed(activities)
        res_emb = self.res_embed(resources)
        cat_emb = torch.cat([act_emb, res_emb], dim=-1)

        # Numerical features
        num_feats = torch.FloatTensor(events_df[['cost', 'time_from_start', 'time_from_previous']].values)
        num_emb = self.num_mlp(num_feats)

        # Combine and project
        combined_emb = torch.cat([cat_emb, num_emb], dim=-1)
        final_emb = self.proj(combined_emb)

        return self.layer_norm(final_emb)
