# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
from data_generator import EpisodeGenerator, collate_batch, SPECIAL_TOKENS, ACTIVITY_VOCAB_SIZE
from model import IOTransformer

# --- Hyperparameters ---
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
NUM_CAT_FEATURES = 2
NUM_NUM_FEATURES = 3
NUM_TIME_FEATURES = 2  # delta_t, absolute_time
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 500
K_SHOTS = 4  # Number of support examples


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Data Generator
    generator = EpisodeGenerator(
        num_cases=500, max_case_len=50,
        num_cat_features=NUM_CAT_FEATURES,
        num_num_features=NUM_NUM_FEATURES
    )

    # 2. Initialize Model
    model = IOTransformer(
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        num_num_features=NUM_NUM_FEATURES, num_time_features=NUM_TIME_FEATURES
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()

        # Randomly select a task for the batch
        task = random.choice(['next_activity', 'remaining_time'])

        # Generate a batch of episodes
        batch_episodes = [generator.create_episode(K_SHOTS, task) for _ in range(BATCH_SIZE)]
        batch = collate_batch(batch_episodes)

        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        optimizer.zero_grad()

        # Forward pass
        activity_logits, time_logits = model(batch)

        # Extract logits only at the query <LABEL> positions
        query_label_logits = activity_logits[batch['loss_mask']] if task == 'next_activity' else time_logits[
            batch['loss_mask']]

        # The target labels need to be shifted to match the head's output vocabulary
        if task == 'next_activity':
            targets = batch['query_true_tokens'] - len(SPECIAL_TOKENS)
        else:  # remaining_time
            targets = batch['query_true_tokens'] - (len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE)

        loss = criterion(query_label_logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Task: {task}, Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "io_transformer.pth")
    print("Training finished and model saved.")


if __name__ == "__main__":
    train()
