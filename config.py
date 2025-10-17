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
            'D_unseen': os.path.join(LOG_DIR, '03_repairExample.xes.gz')
        }
    },

    # --- Core Model Strategy ---
    # Choose 'pretrained' or 'learned'
    # 'pretrained': Uses fixed embeddings from a SentenceTransformer model.
    # 'learned':    Learns embeddings for activities/resources from scratch.
    'embedding_strategy': 'learned',  # <-- CHOOSE YOUR STRATEGY HERE

    # --- Strategy-Specific Parameters ---
    'pretrained_settings': {
        'sbert_model': 'all-MiniLM-L6-v2',
        'embedding_dim': 384,  # Dimension of the sbert model
    },
    'learned_settings': {
        'activity_embedding_dim': 64,
        'resource_embedding_dim': 32,
    },

    # --- Transformer Hyperparameters ---
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 2,
    'dropout': 0.1,
    'num_numerical_features': 3,  # cost, time_from_start, time_from_previous

    # --- Meta-Learning Parameters ---
    'num_shots_range': (2, 8),
    'num_queries': 10,
    'num_shots_test': [1, 5, 10],

    # --- Training Parameters ---
    'lr': 3e-4,
    'epochs': 4,
    'episodes_per_epoch': 200,

    # --- Test Parameters ---
    'num_test_episodes': 200,
    'num_cases_for_testing': 500,
}
