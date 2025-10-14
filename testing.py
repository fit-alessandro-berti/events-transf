# /io_transformer/testing.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error

from utils import build_icl_episode, collate_fn


def evaluate_model(model, test_log, vocabs, k_shots_list=[0, 4, 16], num_test_episodes=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = {}

    for k in k_shots_list:
        print(f"\n--- Evaluating with K={k} shots ---")

        cls_preds, cls_true = [], []
        reg_preds, reg_true = [], []

        for _ in tqdm(range(num_test_episodes), desc=f"Testing K={k}"):
            task = np.random.choice(['next_activity', 'remaining_time'])

            episode = build_icl_episode(task, test_log, vocabs, k_shots=k, is_training=False)
            if episode is None: continue

            # Create a mini-batch of size 1
            batch = collate_fn([episode], vocabs)

            # Move to device
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            with torch.no_grad():
                outputs = model(batch)
                target_mask = batch['target_mask']

                if task == 'next_activity':
                    logits = outputs['classification'][target_mask]
                    pred = torch.argmax(logits, dim=-1).item()
                    cls_preds.append(pred)
                    cls_true.append(batch['targets'].item())

                elif task == 'remaining_time':
                    mu = outputs['regression'][0][target_mask]
                    log_sigma = outputs['regression'][1][target_mask]
                    # Prediction is the mean of the log-normal distribution
                    pred = torch.exp(mu + torch.exp(log_sigma) ** 2 / 2).item()
                    reg_preds.append(pred)
                    reg_true.append(batch['targets'].item())

        # Compute and store metrics
        if cls_true:
            accuracy = accuracy_score(cls_true, cls_preds)
            results[f'{k}-shot_accuracy'] = accuracy
            print(f"Next Activity Accuracy: {accuracy:.4f}")

        if reg_true:
            mae = mean_absolute_error(reg_true, reg_preds)
            results[f'{k}-shot_mae'] = mae
            print(f"Remaining Time MAE: {mae:.4f} hours")

    return results
