# /io_transformer/main.py

import torch
import numpy as np
import random

from data_generator import generate_event_log
from utils import build_vocabularies
from model import IOTransformer
from training import train_model
from testing import evaluate_model

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Configuration ---
DATA_CONFIG = {
    'num_cases': 2000,
    'train_split': 0.8
}

MODEL_CONFIG = {
    'd_model': 128,
    'n_head': 4,
    'n_layers': 4,
    'dropout': 0.1,
}

TRAINING_CONFIG = {
    'epochs': 10,  # Increase for better performance
    'batch_size': 32,
    'lr': 3e-4,
    'episodes_per_epoch': 512,  # Number of ICL episodes to generate per epoch
    'min_k_shots': 1,  # For dynamic K-shot training
    'max_k_shots': 16,
}

TEST_CONFIG = {
    'k_shots_list': [0, 4, 16],
    'num_test_episodes': 500
}


def main():
    print("1. Generating synthetic event log...")
    event_log = generate_event_log(num_cases=DATA_CONFIG['num_cases'])

    print("2. Building vocabularies...")
    vocabs = build_vocabularies(event_log)
    vocab_sizes = {name: len(vocab) for name, vocab in vocabs.items()}
    print(f"Vocabulary sizes: {vocab_sizes}")

    # Split data
    split_idx = int(len(event_log) * DATA_CONFIG['train_split'])
    train_log = event_log[:split_idx]
    test_log = event_log[split_idx:]
    print(f"Data split: {len(train_log)} train cases, {len(test_log)} test cases.")

    print("3. Initializing the IO-Transformer model...")
    model = IOTransformer(
        vocab_sizes=vocab_sizes,
        d_model=MODEL_CONFIG['d_model'],
        n_head=MODEL_CONFIG['n_head'],
        n_layers=MODEL_CONFIG['n_layers'],
        dropout=MODEL_CONFIG['dropout'],
        num_numeric_features=1  # Only 'cost'
    )
    print(f"Model created with ~{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")

    print("\n4. Starting model training...")
    train_model(model, train_log, test_log, vocabs, TRAINING_CONFIG)
    print("Training complete.")

    print("\n5. Evaluating model performance...")
    results = evaluate_model(
        model,
        test_log,
        vocabs,
        k_shots_list=TEST_CONFIG['k_shots_list'],
        num_test_episodes=TEST_CONFIG['num_test_episodes']
    )

    print("\n--- Final Results ---")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()
