# File: config.py
import os

# Define the base directory for logs
LOG_DIR = './logs'

# --- Configuration ---
CONFIG = {
    # ... (data configuration unchanged) ...
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
            'D_unseen': os.path.join(LOG_DIR, '00013_clos2rep.xes.gz')
        }
    },

    # --- 🔻 NEW: Mixture of Experts (MoE) Settings 🔻 ---
    'moe_settings': {
        # Set to 1 to run as a standard (non-MoE) model.
        # Set > 1 to enable the MoE architecture.
        'num_experts': 4
    },
    # --- 🔺 END NEW 🔺 ---

    # --- Core Model Strategy ---
    # ... (embedding_strategy unchanged) ...
    'embedding_strategy': 'learned',

    # ... (strategy-specific parameters unchanged) ...
    'pretrained_settings': {
        'sbert_model': 'all-mpnet-base-v2', # Was: 'all-MiniLM-L6-v2'
        'embedding_dim': 768,               # Was: 384
    },
    'learned_settings': {
        'char_embedding_dim': 64,           # Was: 32
        'char_cnn_output_dim': 128,         # Was: 64
    },

    # ... (transformer hyperparameters unchanged) ...
    'd_model': 256,                 # Was: 128
    'n_heads': 8,                   # Was: 4 (256 / 8 = 32 dim per head)
    'n_layers': 6,                  # Was: 2
    'dropout': 0.15,                # Was: 0.1
    'num_numerical_features': 3,

    # ... (meta-learning parameters unchanged) ...
    'num_shots_range': (1, 20),           # Was: (2, 8)
    'num_queries': 10,
    'num_shots_test': [1, 5, 10, 20],   # Added 20 to align with new range

    # --- Training Parameters ---
    # ... (lr, epochs, etc. unchanged) ...
    'lr': 1e-4,                     # Was: 3e-4
    'epochs': 10,                   # Was: 4, then 10
    'episodes_per_epoch': 1000,      # Was: 200
    'episodic_label_shuffle': 'mixed', # "no" (False), "yes" (True), or "mixed" (alternates each epoch)

    # --- 🔻 NEW: Retrieval-Augmented Training Settings 🔻 ---
    # 'episodic':      Standard K-shot, N-way episodes (default).
    # 'retrieval':     New k-NN-based "hard negative" mining episodes.
    # 'mixed':         Alternates 50/50 between 'episodic' and 'retrieval'.
    'training_strategy': 'mixed',
    'retrieval_train_k': 5,          # k-value for 'retrieval' training episodes
    'retrieval_train_batch_size': 64, # In-batch search space size
    # --- 🔺 END NEW 🔺 ---

    # --- Test Parameters ---
    # 'meta_learning':       Standard episodic evaluation (default).
    # 'retrieval_augmented': Pre-compute all test embeddings, then use
    #                        k-NN for the support set.
    'test_mode': 'retrieval_augmented',
    'test_retrieval_k': [1, 5, 10, 20], # k-values for retrieval_augmented mode
    'test_retrieval_candidate_percentages': [100], # % of candidate pool to sample for k-NN
    'num_test_episodes': 200,
    'num_cases_for_testing': 500,
}

if False:
    # Add all logs in LOG_DIR/out to the training log paths.
    OUT_DIR = os.path.join(LOG_DIR, 'out')
    if os.path.isdir(OUT_DIR):
        # If LOG_DIR/out exists and is not empty, remove the current training logs.
        replaced_training_logs = False
        if os.listdir(OUT_DIR):
            CONFIG['log_paths']['training'] = {}
            replaced_training_logs = True
        out_logs = [
            name for name in os.listdir(OUT_DIR)
            if (name.startswith('log_') or name.startswith('simulated_'))
            and os.path.isfile(os.path.join(OUT_DIR, name))
        ]
        for name in sorted(out_logs):
            CONFIG['log_paths']['training'][name] = os.path.join(OUT_DIR, name)
        if replaced_training_logs:
            print('Replaced training logs with logs from:', OUT_DIR)
            print('Training logs:', CONFIG['log_paths']['training'])
