# main.py
import numpy as np
import torch
import os
import argparse
import shutil  # 🔻 --- NEW IMPORT --- 🔻
import re      # 🔻 --- NEW IMPORT --- 🔻

# --- Import from project files ---
from config import CONFIG
from utils.data_utils import get_task_data
from utils.model_utils import init_loader, create_model
from training import train


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the meta-learning model training script.")

    # Pull defaults directly from the imported CONFIG
    default_config = CONFIG

    # --- 🔻 NEW: Checkpoint and Training Control Arguments 🔻 ---
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help="Directory to save checkpoints and training artifacts."
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="Resume training from the latest checkpoint in --checkpoint_dir."
    )
    parser.add_argument(
        '--stop_after_epoch',
        type=int,
        default=None,
        help="Stop training after this specific epoch number completes (e.g., 1)."
    )
    # --- 🔻 NEW: Cleanup Argument 🔻 ---
    parser.add_argument(
        '--cleanup_checkpoints',
        action='store_true',
        help="Remove all intermediate checkpoints after training, keeping only the last one."
    )
    # --- 🔺 END NEW 🔺 ---

    # --- Core Model Strategy ---
    parser.add_argument(
        '--embedding_strategy',
        type=str,
        default=default_config['embedding_strategy'],
        choices=['learned', 'pretrained'],
        help=f"Embedding strategy to use. (default: {default_config['embedding_strategy']})"
    )
    # ... (rest of model/training args unchanged) ...
    parser.add_argument(
        '--num_experts',
        type=int,
        default=default_config['moe_settings']['num_experts'],
        help=f"Number of experts (MoE > 1). (default: {default_config['moe_settings']['num_experts']})"
    )
    parser.add_argument('--d_model', type=int, default=default_config['d_model'],
                        help=f"Model dimension. (default: {default_config['d_model']})")
    parser.add_argument('--n_heads', type=int, default=default_config['n_heads'],
                        help=f"Number of attention heads. (default: {default_config['n_heads']})")
    parser.add_argument('--n_layers', type=int, default=default_config['n_layers'],
                        help=f"Number of transformer layers. (default: {default_config['n_layers']})")
    parser.add_argument('--dropout', type=float, default=default_config['dropout'],
                        help=f"Dropout rate. (default: {default_config['dropout']})")
    parser.add_argument('--lr', type=float, default=default_config['lr'],
                        help=f"Learning rate. (default: {default_config['lr']})")
    parser.add_argument('--epochs', type=int, default=default_config['epochs'],
                        help=f"Number of epochs. (default: {default_config['epochs']})")
    parser.add_argument(
        '--episodes_per_epoch',
        type=int,
        default=default_config['episodes_per_epoch'],
        help=f"Episodes per epoch. (default: {default_config['episodes_per_epoch']})"
    )
    parser.add_argument(
        '--training_strategy',
        type=str,
        default=default_config['training_strategy'],
        choices=['episodic', 'retrieval', 'mixed'],
        help=f"Training strategy. (default: {default_config['training_strategy']})"
    )
    parser.add_argument(
        '--episodic_label_shuffle',
        type=str,
        default=default_config['episodic_label_shuffle'],
        choices=['no', 'yes', 'mixed'],
        help=f"Episodic label shuffle strategy. (default: {default_config['episodic_label_shuffle']})"
    )
    parser.add_argument(
        '--retrieval_train_k',
        type=int,
        default=default_config['retrieval_train_k'],
        help=f"k-value for retrieval training. (default: {default_config['retrieval_train_k']})"
    )
    parser.add_argument(
        '--num_shots_range',
        type=int,
        nargs=2,
        default=default_config['num_shots_range'],
        help=f"Min and max k-shots for training. (default: {default_config['num_shots_range'][0]} {default_config['num_shots_range'][1]})"
    )
    parser.add_argument(
        '--num_queries',
        type=int,
        default=default_config['num_queries'],
        help=f"Number of queries per class in episodes. (default: {default_config['num_queries']})"
    )

    args = parser.parse_args()

    # --- Update CONFIG with parsed arguments ---
    # This will override the imported defaults for the rest of this script
    CONFIG['embedding_strategy'] = args.embedding_strategy
    CONFIG['moe_settings']['num_experts'] = args.num_experts
    CONFIG['d_model'] = args.d_model
    CONFIG['n_heads'] = args.n_heads
    CONFIG['n_layers'] = args.n_layers
    CONFIG['dropout'] = args.dropout
    CONFIG['lr'] = args.lr
    CONFIG['epochs'] = args.epochs
    CONFIG['episodes_per_epoch'] = args.episodes_per_epoch
    CONFIG['training_strategy'] = args.training_strategy
    CONFIG['episodic_label_shuffle'] = args.episodic_label_shuffle
    CONFIG['retrieval_train_k'] = args.retrieval_train_k
    CONFIG['num_shots_range'] = tuple(args.num_shots_range)  # Convert list back to tuple
    CONFIG['num_queries'] = args.num_queries

    print("--- 🚀 Initializing Training Run with Configuration ---")
    print(f"  - Embedding Strategy: {CONFIG['embedding_strategy']}")
    print(f"  - Num Experts (MoE): {CONFIG['moe_settings']['num_experts']}")
    print(f"  - Training Strategy: {CONFIG['training_strategy']}")
    print(f"  - Epochs: {CONFIG['epochs']}")
    print(f"  - Learning Rate: {CONFIG['lr']}")
    print(f"  - d_model: {CONFIG['d_model']}, n_heads: {CONFIG['n_heads']}, n_layers: {CONFIG['n_layers']}")
    print(f"  - Num Shots Range: {CONFIG['num_shots_range']}")
    # --- 🔻 NEW: Print new args 🔻 ---
    print(f"  - Checkpoint Directory: {args.checkpoint_dir}")
    print(f"  - Resume Training: {args.resume}")
    if args.stop_after_epoch:
        print(f"  - Stop After Epoch: {args.stop_after_epoch}")
    print(f"  - Cleanup Checkpoints: {args.cleanup_checkpoints}")
    # --- 🔺 END NEW 🔺 ---

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    strategy = CONFIG['embedding_strategy']
    print(f"--- Running with embedding strategy: '{strategy}' ---")

    # --- 🔻 MODIFIED: Checkpoint and Resume Logic 🔻 ---
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')
    start_epoch = 0
    latest_checkpoint_path = None

    if args.resume:
        print(f"--- 🔄 Resuming training from {checkpoint_dir} ---")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
        if checkpoints:
            # Find the checkpoint with the highest epoch number
            latest_checkpoint = sorted(checkpoints, key=lambda f: int(re.search(r'(\d+)', f).group(1)))[-1]
            latest_epoch_num = int(re.search(r'(\d+)', latest_checkpoint).group(1))
            start_epoch = latest_epoch_num  # We resume from the *next* epoch
            latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint}. Resuming from epoch {start_epoch + 1}.")
        else:
            print("No checkpoints found. Starting from epoch 1.")
    else:
        print(f"--- 🗑️ Starting new training run. Clearing {checkpoint_dir} ---")
        for filename in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    # --- 🔺 END MODIFIED 🔺 ---

    # --- 1. Data Preparation ---
    print("--- Phase 1: Preparing Training Data ---")
    loader = init_loader(CONFIG)

    # Load artifacts if they exist (e.g., when resuming), otherwise fit
    if args.resume and os.path.exists(artifacts_path):
        print(f"Loading existing artifacts from {artifacts_path}...")
        loader.load_training_artifacts(artifacts_path)
    else:
        print("Fitting new loader and saving artifacts...")
        loader.fit(CONFIG['log_paths']['training'])
        loader.save_training_artifacts(artifacts_path)

    training_logs = loader.transform(CONFIG['log_paths']['training'])

    print("\n--- Phase 2: Creating Training Tasks ---")
    training_tasks = {
        'classification': [get_task_data(log, 'classification') for log in training_logs.values()],
        'regression': [get_task_data(log, 'regression') for log in training_logs.values()]
    }

    # --- 3. Model Initialization ---
    print("\n--- Phase 3: Initializing Model ---")
    model = create_model(CONFIG, loader, device)

    # --- 🔻 NEW: Load weights if resuming 🔻 ---
    if latest_checkpoint_path:
        print(f"Loading model weights from {latest_checkpoint_path}...")
        model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))
    # --- 🔺 END NEW 🔺 ---

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 4. Training ---
    print("\n--- Phase 4: Starting Model Training ---")
    # --- 🔻 MODIFIED: Pass new args to train() 🔻 ---
    train(
        model,
        training_tasks,
        loader,
        CONFIG,
        checkpoint_dir=checkpoint_dir,
        resume_epoch=start_epoch,
        stop_after_epoch=args.stop_after_epoch,
        cleanup_checkpoints=args.cleanup_checkpoints
    )
    # --- 🔺 END MODIFIED 🔺 ---

    print("\n✅ Training complete. Run 'testing.py' to evaluate.")


if __name__ == '__main__':
    main()
