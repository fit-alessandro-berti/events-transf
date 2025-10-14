# /io_transformer/training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from utils import build_icl_episode, collate_fn


class ICLDataset(Dataset):
    """A dataset that generates ICL episodes on the fly."""

    def __init__(self, event_log, vocabs, num_episodes, k_shots_sampler):
        self.event_log = event_log
        self.vocabs = vocabs
        self.num_episodes = num_episodes
        self.k_shots_sampler = k_shots_sampler

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        task = np.random.choice(['next_activity', 'remaining_time'])
        k_shots = self.k_shots_sampler()
        episode = build_icl_episode(task, self.event_log, self.vocabs, k_shots)
        while episode is None:  # Resample if an invalid episode was created
            episode = build_icl_episode(task, self.event_log, self.vocabs, k_shots)
        return episode


def log_normal_nll_loss(mu, log_sigma, targets):
    """Negative log-likelihood loss for a log-normal distribution."""
    # Ensure targets are positive for log
    targets = torch.clamp(targets, min=1e-6)
    log_targets = torch.log(targets)
    sigma = torch.exp(log_sigma)

    loss = (log_targets + log_sigma + 0.5 * np.log(2 * np.pi) +
            0.5 * ((log_targets - mu) / sigma) ** 2)
    return loss.mean()


def train_model(model, train_log, val_log, vocabs, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Loss functions
    cls_criterion = nn.CrossEntropyLoss()

    # DataLoaders
    k_shot_sampler = lambda: np.random.randint(config['min_k_shots'], config['max_k_shots'] + 1)
    train_dataset = ICLDataset(train_log, vocabs, config['episodes_per_epoch'], k_shot_sampler)

    # Use a generator for DataLoader to avoid creating a huge list of episodes in memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocabs)
    )

    for epoch in range(config['epochs']):
        model.train()
        total_cls_loss = 0
        total_reg_loss = 0
        cls_count = 0
        reg_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for batch in pbar:
            optimizer.zero_grad()

            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward pass
            outputs = model(batch)

            # Unpack targets and mask
            targets = batch['targets']
            target_mask = batch['target_mask']  # Shape: (N, S)

            # Separate batches by task type for loss calculation
            # In this simple setup, each batch is mixed. We determine task by target type.
            # A more robust way is to pass task type info in the batch.
            is_cls_task = targets < vocabs['activity'].num_tokens  # Heuristic: reg targets are large floats
            is_reg_task = ~is_cls_task

            loss = 0

            # Classification loss
            if is_cls_task.any():
                cls_logits = outputs['classification'][target_mask & is_cls_task.unsqueeze(1)]
                cls_targets = targets[is_cls_task].long()
                if cls_logits.numel() > 0:
                    cls_loss = cls_criterion(cls_logits, cls_targets)
                    loss += cls_loss
                    total_cls_loss += cls_loss.item()
                    cls_count += 1

            # Regression loss
            if is_reg_task.any():
                mu = outputs['regression'][0][target_mask & is_reg_task.unsqueeze(1)]
                log_sigma = outputs['regression'][1][target_mask & is_reg_task.unsqueeze(1)]
                reg_targets = targets[is_reg_task]
                if mu.numel() > 0:
                    reg_loss = log_normal_nll_loss(mu, log_sigma, reg_targets)
                    loss += reg_loss
                    total_reg_loss += reg_loss.item()
                    reg_count += 1

            if loss != 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            pbar.set_postfix({
                "Cls Loss": f"{total_cls_loss / (cls_count or 1):.3f}",
                "Reg Loss": f"{total_reg_loss / (reg_count or 1):.3f}",
                "LR": f"{scheduler.get_last_lr()[0]:.1e}"
            })

        scheduler.step()
