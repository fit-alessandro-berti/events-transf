
import random
from dataclasses import dataclass
from typing import Optional

import torch


def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: Optional[str] = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_causal_mask(T: int, device: torch.device):
    mask = torch.full((T, T), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # upper triangle is -inf
    return mask


def topk_accuracies(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 3)):
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    res = []
    for k in ks:
        correct_k = correct[:, :k].any(dim=1).float().mean().item()
        res.append(correct_k)
    return res


@dataclass
class EpisodeBatch:
    activity_ids: torch.Tensor
    resource_ids: torch.Tensor
    num_feats: torch.Tensor
    time_feats: torch.Tensor
    token_types: torch.Tensor  # 0=EVENT,1=SPECIAL,2=CLS_LABEL,3=NUM_LABEL,4=PAD
    special_ids: torch.Tensor
    class_label_ids: torch.Tensor
    num_label_values: torch.Tensor
    segment_ids: torch.Tensor  # 0=support, 1=query
    label_pos_mask: torch.Tensor  # [B, T]
    query_label_pos_mask: torch.Tensor  # [B, T]
    y_cls: Optional[torch.Tensor]  # [B]
    y_reg: Optional[torch.Tensor]  # [B]
    attn_mask: Optional[torch.Tensor]  # [T, T]


SPECIAL_TOKENS = {
    "TASK_NEXT_ACTIVITY": 0,
    "TASK_REMAINING_TIME": 1,
    "CASE_SEP": 2,
    "QUERY": 3,
    "LABEL": 4,
    "PAD": 5,
}

TOKEN_TYPES = {
    "EVENT": 0,
    "SPECIAL": 1,
    "CLS_LABEL": 2,
    "NUM_LABEL": 3,
    "PAD": 4,
}
