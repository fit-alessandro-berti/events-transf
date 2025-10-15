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
            'B': os.path.join(LOG_DIR, '02_teleclaims.xes.gz'),
            'C': os.path.join(LOG_DIR, '04_reviewing.xes.gz'),
            'D': os.path.join(LOG_DIR, '03_repairExample.xes.gz'),
        },
        'testing': {
            'D_unseen': os.path.join(LOG_DIR, '08_receipt.xes.gz')
        }
    },

    # --- Model Hyperparameters (Full Size) ---
    'd_model': 128,                 # Increased from 64
    'n_heads': 8,                   # Increased from 4
    'n_layers': 3,                  # Increased from 2
    'dropout': 0.1,
    'num_numerical_features': 3,    # cost, time_from_start, time_from_previous

    # --- Meta-Learning Parameters ---
    'num_shots_range': (2, 10),      # Allow up to 10 shots during training
    'num_queries': 10,
    'num_shots_test': [1, 5, 10],

    # --- Training Parameters (Increased for better performance) ---
    'lr': 3e-4,
    'epochs': 15,                   # Increased from 6
    'episodes_per_epoch': 1000,     # Increased from 200

    # --- Test Parameters ---
    'num_test_episodes': 1000,
    'num_cases_for_testing': 500,
}