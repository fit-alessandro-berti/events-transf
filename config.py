# config.py
import os

# Define the base directory for logs
LOG_DIR = './logs'

# --- Configuration ---
CONFIG = {
    # --- Data Configuration ---
    'log_paths': {
        'training': {
            'A': os.path.join(LOG_DIR, '01_running-example.xes.gz'),
            'B': os.path.join(LOG_DIR, '04_reviewing.xes.gz'),
            'C': os.path.join(LOG_DIR, '08_receipt.xes.gz'),
        },
        'testing': {
            'D_unseen': os.path.join(LOG_DIR, '01_running-example.xes.gz')
        }
    },

    # --- Lightweight Model Hyperparameters (for quick training) ---
    'd_model': 64,                  # Reduced from 128
    'n_heads': 4,                   # Reduced from 8
    'n_layers': 2,                  # Reduced from 3
    'dropout': 0.1,
    'num_numerical_features': 3,    # cost, time_from_start, time_from_previous

    # --- Meta-Learning Parameters ---
    'num_shots_range': (2, 8),      # A slightly smaller range for faster episodes
    'num_queries': 10,
    'num_shots_test': [1, 5, 10],

    # --- Training Parameters (Significantly reduced for speed) ---
    'lr': 3e-4,                     # Slightly higher learning rate can help with shorter training
    'epochs': 5,                    # Reduced from 25
    'episodes_per_epoch': 200,      # Reduced from 1000

    # --- Test Parameters (Reduced for faster evaluation) ---
    'num_test_episodes': 200,       # Reduced from 1000
    'num_cases_for_testing': 500,
}
