# components/prototypical_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalHead(nn.Module):
    """
    Performs prediction using class prototypes or kernel regression.
    This module has no trainable parameters.
    """

    def __init__(self):
        super().__init__()

    def forward_classification(self, support_features, support_labels, query_features):
        """
        Calculates distances to class prototypes.

        Returns:
            torch.Tensor: Negative log-softmax probabilities for the query samples.
        """
        unique_classes = torch.unique(support_labels)
        prototypes = []
        for cls in unique_classes:
            # Calculate mean of support features for each class
            proto = support_features[support_labels == cls].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)  # Shape: (num_classes, d_model)

        # Calculate Euclidean distances from query features to prototypes
        # (batch_size, num_classes)
        distances = torch.cdist(query_features, prototypes)

        # Convert distances to log probabilities (closer is better)
        return F.log_softmax(-distances, dim=1), unique_classes

    def forward_regression(self, support_features, support_labels, query_features, tau=0.1):
        """
        Performs kernel regression.

        Returns:
            torch.Tensor: Predicted regression values for query samples.
        """
        # Calculate squared Euclidean distances: (num_query, num_support)
        distances_sq = torch.cdist(query_features, support_features).pow(2)

        # Compute kernel weights (Gaussian kernel)
        weights = F.softmax(-distances_sq / tau, dim=1)  # Shape: (num_query, num_support)

        # Weighted sum of support labels
        # weights: (nq, ns), support_labels: (ns) -> (nq)
        prediction = torch.mv(weights, support_labels)

        return prediction
