# config.py
import os

# Define the base directory for logs
LOG_DIR = './logs'

# --- Configuration ---
CONFIG = {
    # --- Data Configuration ---
    'log_paths': {
        'training': {
            'o2c': os.path.join(LOG_DIR, '0001_o2c.xes.gz'),
            'hire2retire': os.path.join(LOG_DIR, '00002_hire2retire.xes.gz'),
            'di2re': os.path.join(LOG_DIR, '00003_di2re.xes.gz'),
            'mak2stock': os.path.join(LOG_DIR, '00004_mak2stock.xes.gz'),
            'offer2accept': os.path.join(LOG_DIR, '00005_offer2accept.xes.gz'),
            'quote2order': os.path.join(LOG_DIR, '00006_quote2order.xes.gz'),
            'opp2quote': os.path.join(LOG_DIR, '00007_opp2quote.xes.gz'),
            'lead2opp': os.path.join(LOG_DIR, '00008_lead2opp.xes.gz'),
            'p2p': os.path.join(LOG_DIR, '00009_p2p.xes.gz'),
            'rid2mit': os.path.join(LOG_DIR, '00010_rid2mit.xes.gz'),
            'req2receipt': os.path.join(LOG_DIR, '00011_req2receipt.xes.gz'),
        },
        'testing': {
            'D_unseen': os.path.join(LOG_DIR, '00012_camp2lead.xes.gz')
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
