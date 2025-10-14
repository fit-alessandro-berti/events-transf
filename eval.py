
import json
from typing import Dict

import torch

from losses import regression_loss_at_labels
from utils import topk_accuracies


@torch.no_grad()
def eval_icl(model, builder, device, Ks=(0, 4, 16), episodes: int = 50) -> Dict[str, Dict[str, float]]:
    model.eval()
    report: Dict[str, Dict[str, float]] = {}
    for task in ["cls", "reg"]:
        for K in Ks:
            acc1s, acc3s = [], []
            maes, nlls = [], []
            for _ in range(episodes):
                batch = builder.build_batch(task=task, K=K, batch_size=8, device=device)
                out = model(batch, predict_task="both")
                if task == "cls":
                    mask = batch.query_label_pos_mask.bool()
                    logits_at = out["logits"][mask]
                    targets = batch.y_cls
                    acc1, acc3 = topk_accuracies(logits_at, targets, ks=(1, 3))
                    acc1s.append(acc1); acc3s.append(acc3)
                else:
                    nll, mae, pred = regression_loss_at_labels(out["mu"], out["log_sigma"], batch.query_label_pos_mask, batch.y_reg)
                    maes.append(mae.item()); nlls.append(nll.item())
            key = f"{task.upper()}@{K}-shot"
            if task == "cls":
                report[key] = {"top1": float(sum(acc1s)/len(acc1s)), "top3": float(sum(acc3s)/len(acc3s))}
            else:
                report[key] = {"MAE_min": float(sum(maes)/len(maes)), "NLL": float(sum(nlls)/len(nlls))}
    return report
