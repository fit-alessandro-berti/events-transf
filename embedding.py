# /io_transformer/embedding.py

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LabelEmbedder(nn.Module):
    """Embeds labels for use in ICL support examples."""

    def __init__(self, d_model, regression_bins=None):
        super().__init__()
        self.d_model = d_model
        # For regression, discretize the value and use an embedding layer, which is stable.
        if regression_bins:
            self.reg_emb = nn.Embedding(regression_bins, d_model)
        else:
            # Alternative: A small MLP for continuous values
            self.reg_mlp = nn.Sequential(
                nn.Linear(1, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model)
            )

    def forward(self, labels, task_type):
        if task_type == 'classification':
            # Classification labels are already token IDs, handled by main embedder
            raise NotImplementedError("Classification labels are embedded via the main token embedding layer.")
        elif task_type == 'regression':
            # Unsqueeze to add feature dimension for the MLP
            return self.reg_mlp(labels.unsqueeze(-1))


class EventEmbedder(nn.Module):
    """The main embedding module for event data."""

    def __init__(self, vocab_sizes, d_model=128, num_numeric_features=1):
        super().__init__()
        self.d_model = d_model

        # Embedding for special tokens and categorical features
        self.token_emb = nn.Embedding(vocab_sizes['token'], d_model, padding_idx=0)
        self.activity_emb = nn.Embedding(vocab_sizes['activity'], d_model, padding_idx=0)
        self.resource_emb = nn.Embedding(vocab_sizes['resource'], d_model, padding_idx=0)

        # MLP for numeric features
        self.numeric_mlp = nn.Sequential(
            nn.Linear(num_numeric_features, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # MLP for time features (delta t from start, delta t from previous)
        self.time_mlp = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Final projection layer to combine all embeddings
        self.projection = nn.Linear(d_model * 4, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, batch):
        # Unpack batch dictionary
        tokens = batch['token_ids']
        activities = batch['activity_ids']
        resources = batch['resource_ids']
        numerics = batch['numeric_features']
        times = batch['time_features']

        # Get embeddings
        token_vecs = self.token_emb(tokens)  # For special tokens
        activity_vecs = self.activity_emb(activities)
        resource_vecs = self.resource_emb(resources)
        numeric_vecs = self.numeric_mlp(numerics)
        time_vecs = self.time_mlp(times)

        # For a position with a special token, zero out the feature embeddings
        is_event_mask = (activities > 0).unsqueeze(-1).float()

        # Combine embeddings: sum special token embedding with feature embeddings
        combined = torch.cat([
            activity_vecs * is_event_mask,
            resource_vecs * is_event_mask,
            numeric_vecs * is_event_mask,
            time_vecs * is_event_mask
        ], dim=-1)

        projected_features = self.projection(combined)

        # Add special token embeddings where events are not present
        final_embedding = projected_features + token_vecs

        # Add positional encoding
        # PyTorch Transformer expects (S, N, E)
        final_embedding = self.pos_encoder(final_embedding.permute(1, 0, 2))

        return final_embedding
