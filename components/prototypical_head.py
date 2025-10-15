import torch
import torch.nn as nn
import torch.nn.functional as F

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

        Returns:
            log_probs: (num_query, num_classes)
            unique_classes: tensor of original class ids in the same order as columns in log_probs
        """
        # Normalize features to stabilize distances
        support_features = _l2_normalize(support_features)
        query_features = _l2_normalize(query_features)

        unique_classes = torch.unique(support_labels)
        prototypes = []
        for cls in unique_classes:
            # Mean of support features for each class
            proto = support_features[support_labels == cls].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes, dim=0)  # (num_classes, d_model)
        prototypes = _l2_normalize(prototypes)

        # Cosine similarity logits
        logits = query_features @ prototypes.t()  # (num_query, num_classes)

        return F.log_softmax(logits, dim=1), unique_classes

    def forward_regression(self, support_features, support_labels, query_features, eps: float = 1e-6):
        """
        Kernel regression with per-episode feature whitening + median heuristic bandwidth.

        Returns:
            torch.Tensor: Predicted regression values for query samples. Shape: (num_query,)
        """
        # Center support
        mu = support_features.mean(dim=0, keepdim=True)  # (1, d)
        S = support_features - mu                        # (ns, d)

        # Compute (regularized) covariance and its inverse square root for whitening
        ns, d = S.shape
        # (d, d)
        cov = (S.T @ S) / max(ns - 1, 1)
        cov = cov + (1e-4 * torch.eye(d, device=S.device, dtype=S.dtype))

        try:
            # eig-based inverse sqrt is robust and differentiable
            evals, evecs = torch.linalg.eigh(cov)
            inv_sqrt = evecs @ torch.diag(evals.clamp_min(1e-6).rsqrt()) @ evecs.T
        except RuntimeError:
            # Fallback to cholesky inverse if needed
            L = torch.linalg.cholesky(cov)
            inv_sqrt = torch.cholesky_inverse(L)

        # Whiten support and query features with the same transform
        S_wh = S @ inv_sqrt                              # (ns, d)
        Q_wh = (query_features - mu) @ inv_sqrt          # (nq, d)

        # Squared Euclidean distances in whitened space
        distances_sq = torch.cdist(Q_wh, S_wh).pow(2)    # (nq, ns)

        # Median heuristic for bandwidth (gamma = 1 / median)
        med = torch.median(distances_sq.detach())
        if torch.isfinite(med) and med.item() > 0:
            gamma = 1.0 / (med + eps)
        else:
            # Fallback to mean if median is zero (e.g., identical supports)
            mean_val = distances_sq.detach().mean()
            gamma = 1.0 / (mean_val + eps)

        # Softmax weights over supports
        weights = F.softmax(-gamma * distances_sq, dim=1)  # (nq, ns)

        # Weighted sum of support labels -> predictions
        prediction = weights @ support_labels  # (nq)

        return prediction
