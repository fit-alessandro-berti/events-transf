# File: utils/retrieval_utils.py
import torch
import numpy as np

def find_knn_indices(query_embedding_norm: torch.Tensor,
                     search_embeddings_norm: torch.Tensor,
                     k: int,
                     indices_to_mask: torch.Tensor = None):
    """
    Finds the top-k nearest neighbors using cosine similarity, excluding
    any indices provided in 'indices_to_mask'.

    Args:
        query_embedding_norm (torch.Tensor): Normalized query embedding, shape (1, D).
        search_embeddings_norm (torch.Tensor): Normalized search space, shape (N, D).
        k (int): The number of neighbors to find.
        indices_to_mask (torch.Tensor, optional): A 1D tensor of indices to
                                                  exclude from the search.

    Returns:
        torch.Tensor: A 1D tensor of the top-k indices.
    """
    if k <= 0:
        return torch.tensor([], dtype=torch.long, device=query_embedding_norm.device)

    sims = query_embedding_norm @ search_embeddings_norm.T  # (1, N)

    if indices_to_mask is not None and indices_to_mask.numel() > 0:
        sims[0, indices_to_mask] = -float('inf')

    num_valid = (sims[0] > -float('inf')).sum().item()
    k_to_find = min(k, num_valid)

    if k_to_find <= 0:
        return torch.tensor([], dtype=torch.long, device=sims.device)

    top_k_indices = torch.topk(sims.squeeze(0), k_to_find).indices
    return top_k_indices
