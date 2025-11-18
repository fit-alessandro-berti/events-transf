# File: testing.py
import torch
import numpy as np
import os
import re
import warnings
import argparse
from pathlib import Path  # 🔻 --- NEW IMPORT --- 🔻

# --- Stand-alone execution imports ---
from config import CONFIG
from time_transf import inverse_transform_time
from utils.data_utils import get_task_data
from utils.model_utils import init_loader, create_model, load_model_weights
from evaluation import evaluate_model, evaluate_retrieval_augmented, evaluate_sklearn_baselines, evaluate_pca_knn

if __name__ == '__main__':

    # --- 🔻 MODIFIED: Argument Parsing 🔻 ---
    parser = argparse.ArgumentParser(description="Run the meta-learning model evaluation script.")
    default_config = CONFIG

    # --- 🔻 NEW: Checkpoint Arguments 🔻 ---
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help="Directory to load checkpoints and artifacts from."
    )
    parser.add_argument(
        '--checkpoint_epoch',
        type=int,
        default=None,
        help="Specific epoch checkpoint to test (e.g., 1, 5). Defaults to the latest."
    )
    # --- 🔺 END NEW 🔺 ---

    # --- 🔻 MODIFIED: Updated test_log_name argument 🔻 ---
    parser.add_argument(
        '--test_log_name',
        type=str,
        required=True,  # Make it required
        help="Name of the test log (from config) OR a direct path to a .xes.gz file."
    )
    # --- 🔺 END MODIFIED 🔺 ---
    parser.add_argument(
        '--test_mode',
        type=str,
        default=default_config['test_mode'],
        choices=['meta_learning', 'retrieval_augmented'],
        help=f"Evaluation mode. (default: {default_config['test_mode']})"
    )
    parser.add_argument(
        '--num_test_episodes',
        type=int,
        default=default_config['num_test_episodes'],
        help=f"Number of episodes to run for testing. (default: {default_config['num_test_episodes']})"
    )
    parser.add_argument(
        '--test_retrieval_k',
        type=int,
        nargs='+',
        default=default_config['test_retrieval_k'],
        help=f"List of k-values for retrieval-augmented mode. (default: {default_config['test_retrieval_k']})"
    )
    args = parser.parse_args()

    # --- Update CONFIG with parsed arguments ---
    CONFIG['test_mode'] = args.test_mode
    CONFIG['num_test_episodes'] = args.num_test_episodes
    CONFIG['test_retrieval_k'] = args.test_retrieval_k

    print("--- 🚀 Initializing Test Run with Configuration ---")

    # --- 🔻 NEW: Load saved training config if exists 🔻 ---
    config_path = os.path.join(args.checkpoint_dir, 'training_config.pth')
    if os.path.exists(config_path):
        print(f"Loading training config from {config_path} to match model...")
        saved_config = torch.load(config_path)
        CONFIG['moe_settings'] = saved_config['moe_settings']
        CONFIG['embedding_strategy'] = saved_config['embedding_strategy']
        CONFIG['d_model'] = saved_config['d_model']
        CONFIG['n_heads'] = saved_config['n_heads']
        CONFIG['n_layers'] = saved_config['n_layers']
        CONFIG['dropout'] = saved_config['dropout']
        CONFIG['pretrained_settings'] = saved_config.get('pretrained_settings', CONFIG['pretrained_settings'])
        CONFIG['learned_settings'] = saved_config.get('learned_settings', CONFIG['learned_settings'])
    else:
        print("⚠️ No training config found, using default. This may cause state_dict mismatch.")
    # --- 🔺 END NEW 🔺 ---

    # --- 🔻 NEW: Smart log path resolution 🔻 ---
    log_input = args.test_log_name
    log_path_to_transform = None
    log_key_name = None

    log_path_obj = Path(log_input)

    if log_path_obj.exists() and log_path_obj.is_file():
        print(f"  - Test Log: Found direct path: {log_input}")
        log_path_to_transform = str(log_path_obj.resolve())
        # Get the file name and strip extensions like .xes.gz or .xes
        log_file_name = log_path_obj.name
        log_key_name = re.sub(r'\.xes(\.gz)?$', '', log_file_name, flags=re.IGNORECASE)
    else:
        print(f"  - Test Log: Looking up key in config: {log_input}")
        log_path_to_transform = CONFIG['log_paths']['testing'].get(log_input)
        log_key_name = log_input

    if not log_path_to_transform:
        exit(f"❌ Error: Test log not found. '{log_input}' is not a valid path or config key.")

    if not Path(log_path_to_transform).exists():
        exit(f"❌ Error: Log file not found at resolved path: {log_path_to_transform}")
    # --- 🔺 END NEW 🔺 ---

    print(f"  - Test Mode: {CONFIG['test_mode']}")
    print(f"  - Test Episodes: {CONFIG['num_test_episodes']}")
    print(f"  - Checkpoint Directory: {args.checkpoint_dir}")
    if args.checkpoint_epoch:
        print(f"  - Checkpoint Epoch: {args.checkpoint_epoch}")
    else:
        print("  - Checkpoint Epoch: Latest")
    if CONFIG['test_mode'] == 'retrieval_augmented':
        print(f"  - Retrieval K-values: {CONFIG['test_retrieval_k']}")

    strategy = CONFIG['embedding_strategy']
    print(f"--- Running Testing Script in Stand-Alone Mode (strategy: '{strategy}') ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 🔻 MODIFIED: Use args for paths 🔻 ---
    checkpoint_dir = args.checkpoint_dir
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')
    # --- 🔺 END MODIFIED 🔺 ---

    # --- 🔻 MODIFIED: Load Data 🔻 ---
    print("\n📦 Loading test data...")
    loader = init_loader(CONFIG)
    loader.load_training_artifacts(artifacts_path)

    # The new logic block above has already validated and set these:
    # - log_key_name (the key to use in the dict, e.g., '00013_clos2rep')
    # - log_path_to_transform (the full, valid path to the file)

    log_to_transform = {log_key_name: log_path_to_transform}

    print(f"Transforming log: '{log_key_name}' from {log_path_to_transform}")
    testing_logs = loader.transform(log_to_transform)
    # --- 🔺 END MODIFIED 🔺 ---

    torch.manual_seed(42);
    np.random.seed(42)

    model = create_model(CONFIG, loader, device)

    # --- 🔻 MODIFIED: Update load_model_weights call 🔻 ---
    load_model_weights(
        model,
        checkpoint_dir,
        device,
        epoch_num=args.checkpoint_epoch
    )
    # --- 🔺 END MODIFIED 🔺 ---

    # --- 🔻 MODIFIED: Get correct log 🔻 ---
    unseen_log = testing_logs.get(log_key_name)  # Use the derived key name
    if not unseen_log:
        exit(f"❌ Error: Test log '{log_key_name}' could not be processed.")
    # --- 🔺 END MODIFIED 🔺 ---

    print("\n🛠️ Creating test tasks...")
    # This call MUST return (prefix, label, case_id) tuples
    test_tasks = {
        'classification': get_task_data(unseen_log, 'classification'),
        'regression': get_task_data(unseen_log, 'regression')
    }

    # --- 🔻🔻🔻 MODIFIED: Evaluation Logic 🔻🔻🔻 ---
    # ... (evaluation logic unchanged) ...
    test_mode = CONFIG.get('test_mode', 'meta_learning')
    k_list_meta = CONFIG['num_shots_test']
    k_list_retrieval = CONFIG.get('test_retrieval_k', k_list_meta)

    if test_mode == 'retrieval_augmented':
        print("\n--- Running in Retrieval-Augmented Evaluation Mode ---")
        evaluate_retrieval_augmented(
            model, test_tasks, k_list_retrieval, CONFIG['num_test_episodes']
        )
        print("\n--- Running PCA-kNN Baseline Comparison ---")
        evaluate_pca_knn(
            model, test_tasks, k_list_retrieval, CONFIG['num_test_episodes']
        )

    elif test_mode == 'meta_learning':
        print("\n--- Running in Meta-Learning Evaluation Mode ---")
        evaluate_model(
            model, test_tasks, k_list_meta, CONFIG['num_test_episodes']
        )
        evaluate_sklearn_baselines(
            model, test_tasks, k_list_meta, CONFIG['num_test_episodes']
        )
    else:
        print(f"⚠️ Warning: Unknown test_mode '{test_mode}'. Defaulting to 'meta_learning'.")
        evaluate_model(
            model, test_tasks, k_list_meta, CONFIG['num_test_episodes']
        )
        evaluate_sklearn_baselines(
            model, test_tasks, k_list_meta, CONFIG['num_test_episodes']
        )
    # --- 🔺🔺🔺 END MODIFIED 🔺🔺🔺
