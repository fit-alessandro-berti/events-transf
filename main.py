# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
# NEW: Import the learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_generator import EpisodeGenerator, collate_batch, SPECIAL_TOKENS, ACTIVITY_VOCAB_SIZE
from model import IOTransformer

# --- Hyperparameters ---
# RECOMMENDATION: Increase model capacity and training time
D_MODEL = 256  # Increased from 128
N_LAYERS = 6  # Increased from 4
N_HEADS = 8  # Increased from 4
NUM_CAT_FEATURES = 2
NUM_NUM_FEATURES = 3
NUM_TIME_FEATURES = 2
LEARNING_RATE = 3e-4  # Standard for transformers of this size
BATCH_SIZE = 32  # Increased from 16
NUM_EPOCHS = 2000  # Increased from 500
K_SHOTS = 4


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = EpisodeGenerator(
        num_cases=500, max_case_len=50,
        num_cat_features=NUM_CAT_FEATURES,
        num_num_features=NUM_NUM_FEATURES
    )

    model = IOTransformer(
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        cat_cardinalities=generator.cat_cardinalities,
        num_num_features=NUM_NUM_FEATURES,
        num_time_features=generator.time_feat_dim
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # RECOMMENDATION: Add a learning rate scheduler for better convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()

        task = random.choice(['next_activity', 'remaining_time'])
        batch_episodes = [generator.create_episode(K_SHOTS, task) for _ in range(BATCH_SIZE)]
        batch = collate_batch(batch_episodes)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        optimizer.zero_grad()

        activity_logits, time_logits = model(batch)

        query_label_logits = activity_logits[batch['loss_mask']] if task == 'next_activity' else time_logits[
            batch['loss_mask']]

        if task == 'next_activity':
            targets = batch['query_true_tokens'] - len(SPECIAL_TOKENS)
        else:
            targets = batch['query_true_tokens'] - (len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE)

        loss = criterion(query_label_logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # NEW: Step the scheduler

        if (epoch + 1) % 100 == 0:  # Print less frequently
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Task: {task}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), "io_transformer.pth")
    print("Training finished and model saved.")


if __name__ == "__main__":
    train()