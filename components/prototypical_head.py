# components/prototypical_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


class PrototypicalHead(nn.Module):
    """
    Performs prediction using class prototypes (classification) or
    kernel regression (regression). This module has no trainable parameters.
    """

    def __init__(self):
        super().__init__()

    def forward_classification(self, support_features, support_labels, query_features):
        """
        Calculates log-probabilities for the query samples using cosine-similarity
        prototypes built from support features.
        """
        if support_features.numel() == 0:
            return None, None

        # Normalize features to use cosine similarity
        support_features = _l2_normalize(support_features)
        query_features = _l2_normalize(query_features)

        unique_classes = torch.unique(support_labels)
        prototypes = []
        for cls in unique_classes:
            proto = support_features[support_labels == cls].mean(dim=0)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes, dim=0)
        prototypes = _l2_normalize(prototypes)

        # Cosine similarity logits
        logits = query_features @ prototypes.t()

        return F.log_softmax(logits, dim=1), unique_classes

    def forward_regression(self, support_features, support_labels, query_features, eps: float = 1e-6):
        """
        FIX: Simplified and more robust kernel regression.
        This version avoids matrix inversion (whitening), which can be numerically unstable
        and was the likely source of NaN values. It uses a standard RBF kernel on
        L2 distances with a median heuristic for bandwidth selection.
        """
        if support_features.numel() == 0 or query_features.numel() == 0:
            return torch.zeros(query_features.size(0), device=query_features.device)

        # L2 normalization often helps kernel methods by focusing on direction over magnitude
        support_features_norm = _l2_normalize(support_features)
        query_features_norm = _l2_normalize(query_features)

        # Calculate squared Euclidean distances between each query and all support points
        distances_sq = torch.cdist(query_features_norm, support_features_norm).pow(2)

        # Median heuristic for RBF kernel bandwidth (gamma)
        with torch.no_grad():
            # Detach to not influence gradients through the bandwidth calculation
            median_dist = torch.median(distances_sq)

        # Fallback if median is zero or non-finite (e.g., all support points are identical)
        if not torch.isfinite(median_dist) or median_dist <= 0:
            median_dist = distances_sq.mean()

        # Add epsilon for numerical stability
        gamma = 1.0 / (median_dist + eps)

        # Calculate kernel weights using softmax (ensures weights sum to 1)
        # This is equivalent to an RBF kernel: exp(-gamma * dist^2)
        weights = F.softmax(-gamma * distances_sq, dim=1)

        # Prediction is the weighted average of the support labels
        prediction = weights @ support_labels.view(-1)

        return prediction
