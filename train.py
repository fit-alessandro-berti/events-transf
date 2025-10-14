from dataclasses import dataclass

import torch
from torch.optim import AdamW

from io_transformer import IOTransformer
from losses import classification_loss_at_labels, regression_loss_at_labels, lognormal_nll


@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 2000
    grad_clip: float = 1.0
    dropout: float = 0.1


class Trainer:
    def __init__(self, model: IOTransformer, device: torch.device, cfg: TrainConfig):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.step = 0

    def train_icl_step(self, batch, task: str):
        self.model.train()
        out = self.model(batch, predict_task="both")
        losses = {}
        if task == "cls":
            loss, logits_at = classification_loss_at_labels(out["logits"], batch.query_label_pos_mask, batch.y_cls)
            losses["cls"] = loss
        else:
            nll, mae, pred = regression_loss_at_labels(out["mu"], out["log_sigma"], batch.query_label_pos_mask,
                                                       batch.y_reg)
            losses["reg_nll"] = nll

        total_loss = sum(losses.values())
        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.step += 1
        return {k: v.item() for k, v in losses.items()}

    def train_causal_step(self, batch):
        self.model.train()
        out = self.model(batch, predict_task="both")
        logits = out["logits"][:, :-1, :]
        mu = out["mu"][:, :-1]
        log_sigma = out["log_sigma"][:, :-1]

        B, T_minus_1, C = logits.shape
        loss_cls = torch.nn.functional.cross_entropy(
            logits.reshape(-1, C),
            batch.next_cls_targets.reshape(-1),
            reduction="mean",
            ignore_index=-100,
        )

        # Mask out padded values for regression loss
        reg_mask = (batch.next_cls_targets != -100)
        y = batch.next_reg_targets[reg_mask]
        mu_m = mu[reg_mask]
        log_sigma_m = log_sigma[reg_mask]

        if y.numel() > 0:
            loss_reg = (0.5 * (((torch.log(y.clamp_min(1e-6)) - mu_m) / torch.exp(
                log_sigma_m)) ** 2) + log_sigma_m + torch.log(y.clamp_min(1e-6))).mean()
        else:
            loss_reg = torch.tensor(0.0, device=self.device)

        total_loss = 0.5 * loss_cls + 0.5 * loss_reg
        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.step += 1
        return {"causal_cls": loss_cls.item(), "causal_reg": loss_reg.item()}
