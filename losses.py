
import torch
import torch.nn.functional as F


def lognormal_nll(y: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    y = torch.clamp(y, min=eps)
    sigma = torch.exp(log_sigma)
    return 0.5 * (((torch.log(y) - mu) / sigma) ** 2) + log_sigma + torch.log(y)


def classification_loss_at_labels(logits: torch.Tensor, label_pos_mask: torch.Tensor, targets: torch.Tensor):
    B, T, C = logits.size()
    mask = label_pos_mask.bool()
    logits_at = logits[mask]
    loss = F.cross_entropy(logits_at, targets, reduction='mean')
    return loss, logits_at


def regression_loss_at_labels(mu: torch.Tensor, log_sigma: torch.Tensor, label_pos_mask: torch.Tensor, targets: torch.Tensor):
    mask = label_pos_mask.bool()
    mu_at = mu[mask]
    log_sigma_at = log_sigma[mask]
    y = targets
    nll = lognormal_nll(y, mu_at, log_sigma_at).mean()
    pred = torch.exp(mu_at + 0.5 * (torch.exp(log_sigma_at) ** 2))
    mae = torch.abs(pred - y).mean()
    return nll, mae, pred
