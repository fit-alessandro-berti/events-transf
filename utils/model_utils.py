# File: utils/model_utils.py
import torch
import os
import re

# --- Import from project files ---
from config import CONFIG
from data_generator import XESLogLoader
from components.meta_learner import MetaLearner


def init_loader(config):
    """Initializes and returns an XESLogLoader based on the config."""
    strategy = config['embedding_strategy']
    sbert_model_name = config['pretrained_settings']['sbert_model']
    loader = XESLogLoader(strategy=strategy, sbert_model_name=sbert_model_name)
    return loader


def create_model(config, loader, device):
    """
    Initializes the MetaLearner model based on the config and loader.
    """
    strategy = config['embedding_strategy']

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

    model = MetaLearner(
        strategy=strategy, num_feat_dim=config['num_numerical_features'],
        d_model=config['d_model'], n_heads=config['n_heads'],
        n_layers=config['n_layers'], dropout=config['dropout'], **model_params
    ).to(device)

    # Pass the character vocabulary to the model
    if strategy == 'learned':
        model.set_char_vocab(loader.char_to_id)

    return model


def load_model_weights(model, checkpoint_dir, device):
    """Finds the latest checkpoint and loads its weights into the model."""
    if not os.path.isdir(checkpoint_dir):
        exit("❌ Error: Checkpoint directory not found.")

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        exit("❌ Error: No model checkpoints found.")

    latest_checkpoint_path = os.path.join(
        checkpoint_dir,
        sorted(checkpoints, key=lambda f: int(re.search(r'(\d+)', f).group(1)))[-1]
    )
    print(f"🔍 Found latest checkpoint: {os.path.basename(latest_checkpoint_path)}")

    print(f"💾 Loading weights from {latest_checkpoint_path}...")
    model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))
    return latest_checkpoint_path
