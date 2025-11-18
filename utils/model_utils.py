# File: utils/model_utils.py
import torch
import os
import re

# --- Import from project files ---
from config import CONFIG
from data_generator import XESLogLoader
# --- 🔻 MODIFIED IMPORTS 🔻 ---
from components.meta_learner import MetaLearner
from components.moe_model import MoEModel  # Import the new MoE wrapper


# --- 🔺 END MODIFIED 🔺 ---


def init_loader(config):
    """Initializes and returns an XESLogLoader based on the config."""
    # ... (function unchanged) ...
    strategy = config['embedding_strategy']
    sbert_model_name = config['pretrained_settings']['sbert_model']
    loader = XESLogLoader(strategy=strategy, sbert_model_name=sbert_model_name)
    return loader


def create_model(config, loader, device):
    """
    Initializes the MoEModel (which wraps MetaLearner(s))
    based on the config and loader.
    """
    # ... (function unchanged) ...
    strategy = config['embedding_strategy']

    # --- 🔻 MoE Config 🔻 ---
    moe_config = config.get('moe_settings', {})
    num_experts = moe_config.get('num_experts', 1)
    # --- 🔺 End MoE Config 🔺 ---

    if strategy == 'pretrained':
        model_params = {'embedding_dim': config['pretrained_settings']['embedding_dim']}
    else:  # learned
        # Ensure loader is fitted or artifacts loaded to get char_vocab_size
        if not loader.char_to_id:
            raise RuntimeError("Loader must be fitted or artifacts loaded before creating 'learned' model.")
        model_params = {
            'char_vocab_size': len(loader.char_to_id),
            'char_embedding_dim': config['learned_settings']['char_embedding_dim'],
            'char_cnn_output_dim': config['learned_settings']['char_cnn_output_dim'],
        }

    # --- 🔻 MODIFIED: Instantiate MoEModel 🔻 ---
    # This will create MetaLearner(s) internally
    model = MoEModel(
        num_experts=num_experts,
        strategy=strategy,
        num_feat_dim=config['num_numerical_features'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        **model_params
    ).to(device)
    # --- 🔺 END MODIFIED 🔺 ---

    # Pass the character vocabulary to the model
    # The MoEModel will pass it down to all its experts
    if strategy == 'learned':
        model.set_char_vocab(loader.char_to_id)

    return model


# --- 🔻 MODIFIED: Function signature and logic updated 🔻 ---
def load_model_weights(model, checkpoint_dir, device, epoch_num=None):
    """
    Finds the latest checkpoint (or a specific one) and loads its weights.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_dir (str): The directory containing checkpoints.
        device (torch.device): The device to map weights to.
        epoch_num (int, optional): Specific epoch to load. If None, loads latest.
    """
    if not os.path.isdir(checkpoint_dir):
        exit(f"❌ Error: Checkpoint directory not found at {checkpoint_dir}")

    checkpoint_path = None

    if epoch_num is not None:
        # --- Load specific epoch ---
        checkpoint_name = f"model_epoch_{epoch_num}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            exit(f"❌ Error: Specific checkpoint not found: {checkpoint_path}")
        print(f"🔍 Found specific checkpoint: {checkpoint_name}")
    else:
        # --- Load latest epoch (original logic) ---
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
        if not checkpoints:
            exit(f"❌ Error: No model checkpoints found in {checkpoint_dir}.")

        latest_checkpoint_name = sorted(checkpoints, key=lambda f: int(re.search(r'(\d+)', f).group(1)))[-1]
        checkpoint_path = os.path.join(
            checkpoint_dir,
            latest_checkpoint_name
        )
        print(f"🔍 Found latest checkpoint: {latest_checkpoint_name}")

    print(f"💾 Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return checkpoint_path
# --- 🔺 END MODIFIED 🔺 ---
