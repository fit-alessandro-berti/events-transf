# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_generator import EpisodeGenerator, collate_batch, SPECIAL_TOKENS, ACTIVITY_VOCAB_SIZE
from model import IOTransformer

# --- Hyperparameters ---
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8

# [activity, resource, group]
NUM_CAT_FEATURES = 3
# [amount, amount_norm, (padding)]
NUM_NUM_FEATURES = 3
NUM_TIME_FEATURES = 2

LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_EPOCHS = 800
K_SHOTS = 4

# Loss weights
AUX_SUPPORT_LOSS_WEIGHT = 1.0   # support <LABEL> positions
DENSE_LOSS_WEIGHT = 1.0         # dense per-<EVENT> positions


def train():
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = EpisodeGenerator(
        num_cases=1200,
        max_case_len=50,
        num_cat_features=NUM_CAT_FEATURES,
        num_num_features=NUM_NUM_FEATURES,
        n_models=4,
    )

    model = IOTransformer(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        cat_cardinalities=generator.cat_cardinalities,
        num_num_features=NUM_NUM_FEATURES,
        num_time_features=generator.time_feat_dim
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    # a bit of label smoothing helps generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()

        # Mix tasks WITHIN a batch for multitask stability
        batch_episodes = []
        for _ in range(BATCH_SIZE):
            task = random.choice(['next_activity', 'remaining_time'])
            batch_episodes.append(generator.create_episode(K_SHOTS, task))

        batch = collate_batch(batch_episodes)

        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        optimizer.zero_grad()

        activity_logits, time_logits = model(batch)
        tokens = batch['tokens']                           # [B,T]
        loss_mask = batch['loss_mask']                     # [B,T] True at query <LABEL>
        tasks = batch['tasks']                             # list length B
        B, T = tokens.shape

        # Build per-sample task masks
        is_next = torch.tensor([1 if t == 'next_activity' else 0 for t in tasks],
                               device=device, dtype=torch.bool)  # [B]
        is_time = ~is_next

        # ===== Query loss at the query <LABEL> positions (final token per sample) =====
        query_mask_next = loss_mask & is_next.unsqueeze(1)  # [B,T]
        query_mask_time = loss_mask & is_time.unsqueeze(1)  # [B,T]

        query_logits_next = activity_logits[query_mask_next]        # [Nq_next, |A|]
        query_logits_time = time_logits[query_mask_time]            # [Nq_time, |B|]

        q_targets_all = batch['query_true_tokens']  # [B]
        if query_logits_next.numel() > 0:
            targets_next = (q_targets_all - len(SPECIAL_TOKENS))[is_next]  # [Nq_next]
            query_loss_next = criterion(query_logits_next, targets_next)
        else:
            query_loss_next = torch.tensor(0.0, device=device)

        if query_logits_time.numel() > 0:
            targets_time = (q_targets_all - (len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE))[is_time]  # [Nq_time]
            query_loss_time = criterion(query_logits_time, targets_time)
        else:
            query_loss_time = torch.tensor(0.0, device=device)

        query_loss = query_loss_next + query_loss_time

        # ===== Auxiliary support loss at ALL support <LABEL> positions =====
        label_token_id = SPECIAL_TOKENS['<LABEL>']
        label_positions = (tokens == label_token_id)                   # [B,T]
        support_label_mask = label_positions & (~loss_mask)            # [B,T]

        next_token_ids = torch.roll(tokens, shifts=-1, dims=1)         # [B,T]

        support_mask_next = support_label_mask & is_next.unsqueeze(1)
        support_mask_time = support_label_mask & is_time.unsqueeze(1)

        support_logits_next = activity_logits[support_mask_next]       # [Ns_next, |A|]
        support_logits_time = time_logits[support_mask_time]           # [Ns_time, |B|]

        if support_logits_next.numel() > 0:
            support_targets_next = (next_token_ids - len(SPECIAL_TOKENS))[support_mask_next]
            support_loss_next = criterion(support_logits_next, support_targets_next)
        else:
            support_loss_next = torch.tensor(0.0, device=device)

        if support_logits_time.numel() > 0:
            support_targets_time = (next_token_ids - (len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE))[support_mask_time]
            support_loss_time = criterion(support_logits_time, support_targets_time)
        else:
            support_loss_time = torch.tensor(0.0, device=device)

        support_loss = support_loss_next + support_loss_time

        # ===== Dense per-<EVENT> supervision =====
        if 'dense_next_targets' in batch:
            dense_mask_next = batch['dense_mask_next']  # [B,T] bool
            dense_mask_time = batch['dense_mask_time']  # [B,T] bool

            dense_logits_next = activity_logits[dense_mask_next]        # [Nd_next, |A|]
            dense_logits_time = time_logits[dense_mask_time]            # [Nd_time, |B|]

            if dense_logits_next.numel() > 0:
                dense_targets_next = batch['dense_next_targets'][dense_mask_next]  # already [0..|A|-1]
                dense_loss_next = criterion(dense_logits_next, dense_targets_next)
            else:
                dense_loss_next = torch.tensor(0.0, device=device)

            if dense_logits_time.numel() > 0:
                dense_targets_time = batch['dense_time_targets'][dense_mask_time]  # already [0..|B|-1]
                dense_loss_time = criterion(dense_logits_time, dense_targets_time)
            else:
                dense_loss_time = torch.tensor(0.0, device=device)

            dense_loss = dense_loss_next + dense_loss_time
        else:
            dense_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = query_loss + AUX_SUPPORT_LOSS_WEIGHT * support_loss + DENSE_LOSS_WEIGHT * dense_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                n_q_next = int(query_mask_next.sum().item())
                n_q_time = int(query_mask_time.sum().item())
                n_s_next = int(support_mask_next.sum().item())
                n_s_time = int(support_mask_time.sum().item())
                if 'dense_mask_next' in batch:
                    n_d_next = int(batch['dense_mask_next'].sum().item())
                    n_d_time = int(batch['dense_mask_time'].sum().item())
                else:
                    n_d_next = n_d_time = 0
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Loss: {loss.item():.4f} | Q(next:{n_q_next}, time:{n_q_time})={query_loss.item():.4f} "
                f"| S(next:{n_s_next}, time:{n_s_time})={support_loss.item():.4f} "
                f"| D(next:{n_d_next}, time:{n_d_time})={dense_loss.item():.4f} "
                f"| LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    torch.save(model.state_dict(), "io_transformer.pth")
    print("Training finished and model saved.")


if __name__ == "__main__":
    train()
