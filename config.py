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
    # 'pretrained': Uses fixed embeddings from a SentenceTransformer model.
    # 'learned':    Learns embeddings from scratch using a character-level CNN.
    'embedding_strategy': 'learned',

    # --- Strategy-Specific Parameters ---
    'pretrained_settings': {
        'sbert_model': 'all-MiniLM-L6-v2',
        'embedding_dim': 384,
    },
    'learned_settings': {
        'char_embedding_dim': 32,
        'char_cnn_output_dim': 64,
    },

    # --- Transformer Hyperparameters ---
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 2,
    'dropout': 0.1,
    'num_numerical_features': 3,

    # --- Meta-Learning Parameters ---
    'num_shots_range': (2, 8),
    'num_queries': 10,
    'num_shots_test': [1, 5, 10],

    # --- Training Parameters ---
    'lr': 3e-4,
    'epochs': 4,
    'episodes_per_epoch': 200,
    'episodic_label_shuffle': False, # <-- NEW: Set to True to enable label shuffling augmentation

    # --- Test Parameters ---
    'num_test_episodes': 200,
    'num_cases_for_testing': 500,
}
