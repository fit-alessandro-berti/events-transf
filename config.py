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
    # Suggestions to "beef up" the model:
    # 1. Use a larger, more powerful SentenceTransformer model (e.g., all-mpnet-base-v2).
    # 2. Increase the corresponding embedding_dim.
    'pretrained_settings': {
        'sbert_model': 'all-mpnet-base-v2', # Was: 'all-MiniLM-L6-v2'
        'embedding_dim': 768,               # Was: 384
    },
    # 3. Increase the character embedding and CNN dimensions for the 'learned' strategy
    #    to better match the increased d_model.
    'learned_settings': {
        'char_embedding_dim': 64,           # Was: 32
        'char_cnn_output_dim': 128,         # Was: 64
    },

    # --- Transformer Hyperparameters ---
    # 4. Increase the core model dimensions:
    #    - d_model: The main embedding size of the transformer. Larger = more capacity.
    #    - n_heads: Number of attention heads. Must be a divisor of d_model.
    #    - n_layers: The number of transformer blocks (depth). Deeper = more complex.
    #    - dropout: Slightly increase dropout to regularize the larger model.
    'd_model': 256,                 # Was: 128
    'n_heads': 8,                   # Was: 4 (256 / 8 = 32 dim per head)
    'n_layers': 6,                  # Was: 2
    'dropout': 0.15,                # Was: 0.1
    'num_numerical_features': 3,

    # --- Meta-Learning Parameters ---
    # User requested change: 1 to 20 shots
    'num_shots_range': (1, 20),           # Was: (2, 8)
    'num_queries': 10,
    'num_shots_test': [1, 5, 10, 20],   # Added 20 to align with new range

    # --- Training Parameters ---
    # 5. A larger model often requires more training (epochs, episodes)
    #    and a smaller, more stable learning rate.
    # User requested change: Increase epochs
    'lr': 1e-4,                     # Was: 3e-4
    'epochs': 15,                   # Was: 4, then 10
    'episodes_per_epoch': 500,      # Was: 200
    'episodic_label_shuffle': True, # Set to True to enable label shuffling augmentation

    # --- Test Parameters ---
    'num_test_episodes': 200,
    'num_cases_for_testing': 500,
}