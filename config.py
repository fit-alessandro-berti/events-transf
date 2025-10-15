# --- Configuration ---
CONFIG = {
    # Model Hyperparameters (Increased capacity)
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 3,
    'dropout': 0.1,
    'num_numerical_features': 3,  # cost, time_from_start, time_from_previous

    # Meta-Learning Parameters (More robust episodes)
    'num_shots_range': (3, 10),  # This range is fine, but could be widened to (2, 15)
    'num_queries': 10,
    'num_shots_test': [1, 5, 10],

    # Training Parameters (Significantly more training)
    'lr': 1e-4,
    'epochs': 25,
    'episodes_per_epoch': 1000,

    # Increase test episodes for more stable evaluation
    'num_test_episodes': 1000,
}
