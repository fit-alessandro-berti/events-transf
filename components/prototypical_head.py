# components/prototypical_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Performs L2 normalization on the last dimension of a tensor."""
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


class PrototypicalHead(nn.Module):
    """
    Performs prediction using class prototypes (classification) or
    kernel regression (regression). This module has almost no trainable params.

    Updates:
    - Start with a slightly lower logit temperature init (5.0) to avoid
      over-sharp logits early (helps classification stability with smoothing).
    - Adaptive prototype shrinkage by support count: shrink more for 1-shot,
      less for 5/10-shot. Keeps prototypes discriminative when data is richer.
    """

    def __init__(self, init_logit_scale: float = 5.0):
        super().__init__()
        # Learnable temperature for cosine logits (helps early convergence/calibration).
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        # Scalar controls base shrinkage; sigmoid -> (0,1). We'll cap in code to <= 0.4.
        self._proto_shrink = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ~ 0.119 (mild by default)

    def forward_classification(self, support_features, support_labels, query_features):
        """
        Calculates logits for the query samples using cosine-similarity
        prototypes built from support features (with adaptive shrinkage).
        """
        if support_features.numel() == 0:
            return None, None

        # Normalize features to use cosine similarity, which is often more stable
        support_features = _l2_normalize(support_features)
        query_features = _l2_normalize(query_features)

        unique_classes = torch.unique(support_labels)
        class_means = []
        class_counts = []
        for cls in unique_classes:
            idx = (support_labels == cls)
            class_counts.append(idx.sum())
            proto = support_features[idx].mean(dim=0)
            class_means.append(proto)
        class_means = torch.stack(class_means, dim=0)          # (C, D)
        counts = torch.stack(class_counts).float().clamp_min(1)  # (C,)

        # Global centroid across all support
        global_centroid = support_features.mean(dim=0, keepdim=True)  # (1, D)

        # Adaptive shrinkage per class: alpha_c = alpha_base / sqrt(n_c), capped
        alpha_base = torch.sigmoid(self._proto_shrink).clamp(0.0, 0.4)
        alpha_per_class = (alpha_base / counts.sqrt()).unsqueeze(1)  # (C,1)
        prototypes = (1.0 - alpha_per_class) * class_means + alpha_per_class * global_centroid

        prototypes = _l2_normalize(prototypes)  # Re-normalize the final prototypes

        # Cosine similarity is a dot product with normalized vectors.
        # Apply learnable temperature to control margin/sharpness of logits.
        scale = self.logit_scale.clamp(1.0, 100.0)
        logits = (query_features @ prototypes.t()) * scale

        return logits, unique_classes

    def forward_regression(self, support_features, support_labels, query_features, eps: float = 1e-6):
        """
        Simplified and robust kernel regression with an RBF-like weighting via softmax.
        """
        if support_features.numel() == 0 or query_features.numel() == 0:
            return torch.zeros(query_features.size(0), device=query_features.device)

        # L2 normalization often helps kernel methods by focusing on direction over magnitude
        support_features_norm = _l2_normalize(support_features)
        query_features_norm = _l2_normalize(query_features)

        # Calculate squared Euclidean distances between each query and all support points
        distances_sq = torch.cdist(query_features_norm, support_features_norm).pow(2)

        # Median heuristic for RBF kernel bandwidth (gamma), a robust method
        with torch.no_grad():
            median_dist = torch.median(distances_sq.detach())

        # Fallback if median is zero (e.g., all support points are identical)
        if not torch.isfinite(median_dist) or median_dist <= 0:
            median_dist = distances_sq.mean()

        gamma = 1.0 / (median_dist + eps)

        # Calculate kernel weights using softmax, ensures weights sum to 1. weights = exp(-gamma * dist^2)
        weights = F.softmax(-gamma * distances_sq, dim=1)

        # Prediction is the weighted average of the support labels
        prediction = weights @ support_labels.view(-1)

        return prediction
