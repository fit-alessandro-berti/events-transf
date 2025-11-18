# File: testing.py
import torch
import numpy as np
import os
import re
import warnings
import argparse

# --- Stand-alone execution imports ---
from config import CONFIG
from time_transf import inverse_transform_time
from utils.data_utils import get_task_data
from utils.model_utils import init_loader, create_model, load_model_weights
# 🔻 --- MODIFIED IMPORTS --- 🔻
from evaluation import evaluate_model, evaluate_retrieval_augmented, evaluate_sklearn_baselines, evaluate_pca_knn

# 🔺 --- END MODIFIED --- 🔺


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

    available_test_logs = list(default_config['log_paths']['testing'].keys())
    default_test_log = available_test_logs[0] if available_test_logs else None
    parser.add_argument(
        '--test_log_name',
        type=str,
        default=default_test_log,
        choices=available_test_logs,
        help=f"Name of the test log to evaluate. (default: {default_test_log})"
    )
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
    print(f"  - Test Log: {args.test_log_name}")
    print(f"  - Test Mode: {CONFIG['test_mode']}")
    print(f"  - Test Episodes: {CONFIG['num_test_episodes']}")
    # --- 🔻 NEW: Print new args 🔻 ---
    print(f"  - Checkpoint Directory: {args.checkpoint_dir}")
    if args.checkpoint_epoch:
        print(f"  - Checkpoint Epoch: {args.checkpoint_epoch}")
    else:
        print("  - Checkpoint Epoch: Latest")
    # --- 🔺 END NEW 🔺 ---
    if CONFIG['test_mode'] == 'retrieval_augmented':
        print(f"  - Retrieval K-values: {CONFIG['test_retrieval_k']}")
    # --- 🔺 END MODIFIED 🔺 ---

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
    test_log_name = args.test_log_name
    if not test_log_name:
        exit("❌ Error: No test log specified or found in config.")
    log_path = CONFIG['log_paths']['testing'].get(test_log_name)
    if not log_path:
        exit(f"❌ Error: Test log key '{test_log_name}' not found in CONFIG['log_paths']['testing'].")
    log_to_transform = {test_log_name: log_path}
    print(f"Transforming log: '{test_log_name}' from {log_path}")
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
    unseen_log = testing_logs.get(test_log_name)
    if not unseen_log:
        exit(f"❌ Error: Test log '{test_log_name}' could not be processed.")
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
