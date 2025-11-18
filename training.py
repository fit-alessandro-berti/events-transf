# File: training.py
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from torch.cuda.amp import GradScaler  # 🔻 --- MODIFIED IMPORT --- 🔻

# --- Import from project files ---
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_generator import XESLogLoader
from training_strategies.episodic_strategy import run_episodic_step
from training_strategies.retrieval_strategy import run_retrieval_step
from training_strategies.train_utils import evaluate_embedding_quality


def train(model, training_tasks, loader, config, checkpoint_dir, resume_epoch=0, stop_after_epoch=None):
    """
    Main training loop.
    ... (docstring unchanged) ...
    """
    print(f"🚀 Starting meta-training...")
    if resume_epoch > 0:
        print(f"--- Resuming from epoch {resume_epoch + 1} ---")

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    # --- 🔻 MODIFIED: Scheduler Initialization Fix 🔻 ---
    # 1. Initialize the scheduler with default last_epoch=-1
    # This forces it to correctly populate 'initial_lr' in the optimizer param_groups.
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    # 2. If resuming, *now* we manually set the last_epoch.
    # This avoids the KeyError because 'initial_lr' is already set.
    if resume_epoch > 0:
        scheduler.last_epoch = resume_epoch
    # --- 🔺 END MODIFIED 🔺 ---

    # --- 🔻 NEW: Initialize GradScaler for AMP 🔻 ---
    # Only enable scaler if on CUDA
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    print(f"✅ Automatic Mixed Precision (AMP) enabled: {use_amp}")
    # --- 🔺 END NEW 🔺 ---

    cls_task_pools = [pool for pool in training_tasks['classification'] if pool]
    reg_task_pools = [pool for pool in training_tasks['regression'] if pool]

    if not cls_task_pools and not reg_task_pools:
        print("❌ Error: No valid training tasks available. Aborting training.")
        return

    # --- Training Strategy Setup ---
    training_strategy = config.get('training_strategy', 'episodic')
    print(f"✅ Training Strategy: '{training_strategy}'")
    if training_strategy in ['retrieval', 'mixed']:
        print(f"  - Retrieval k (train): {config.get('retrieval_train_k', 5)}")
        print(f"  - Retrieval batch size (train): {config.get('retrieval_train_batch_size', 64)}")

    shuffle_strategy = str(config.get('episodic_label_shuffle', 'no')).lower()
    print(f"✅ Episodic Label Shuffle strategy set to: '{shuffle_strategy}'")

    # --- MoE Setup ---
    num_experts = model.num_experts
    if num_experts > 1:
        print(f"✅ MoE Training enabled: Randomly selecting 1 of {num_experts} experts per step.")

    # --- 🔻 MODIFIED: Loop starts from resume_epoch 🔻 ---
    for epoch in range(resume_epoch, config['epochs']):
        # --- 🔺 END MODIFIED 🔺 ---
        model.train()  # Set the main MoEModel to train mode
        total_loss = 0.0

        should_shuffle_labels = False
        if shuffle_strategy == 'yes':
            should_shuffle_labels = True
        elif shuffle_strategy == 'mixed':
            should_shuffle_labels = (epoch % 2 == 0)

        epoch_desc = f"Epoch {epoch + 1}/{config['epochs']}"
        if shuffle_strategy != 'no':
            epoch_desc += f" (Shuffle: {'ON' if should_shuffle_labels else 'OFF'})"

        progress_bar = tqdm(range(config['episodes_per_epoch']), desc=epoch_desc)

        for step in progress_bar:

            # --- MoE Expert Selection ---
            expert_to_train_id = random.randint(0, num_experts - 1)
            active_expert = model.experts[expert_to_train_id]

            # --- Determine which strategy to use for this step ---
            current_train_mode = training_strategy
            if training_strategy == 'mixed':
                current_train_mode = 'retrieval' if step % 2 == 0 else 'episodic'

            # --- Pick task type and data pool (common to both strategies) ---
            task_type = random.choice(['classification', 'regression'])
            if task_type == 'classification' and cls_task_pools:
                task_data_pool = random.choice(cls_task_pools)
            elif task_type == 'regression' and reg_task_pools:
                task_data_pool = random.choice(reg_task_pools)
            else:
                task_type = 'regression' if reg_task_pools else 'classification'
                task_data_pool = random.choice(reg_task_pools if reg_task_pools else cls_task_pools)

            if not task_data_pool: continue

            optimizer.zero_grad(set_to_none=True)

            # 🔻🔻🔻 REFACTORED STEP LOGIC (with AMP) 🔻🔻🔻
            loss = None
            progress_bar_task = "skip"

            # --- 🔻 MODIFIED: Use new torch.amp.autocast syntax 🔻 ---
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                if current_train_mode == 'episodic':
                    loss, progress_bar_task = run_episodic_step(
                        active_expert,  # <-- MODIFIED
                        task_data_pool,
                        task_type,
                        config,
                        should_shuffle_labels
                    )
                elif current_train_mode == 'retrieval':
                    loss, progress_bar_task = run_retrieval_step(
                        active_expert,  # <-- MODIFIED
                        task_data_pool,
                        task_type,
                        config
                    )
            # --- 🔺 END MODIFIED 🔺 ---
            # 🔺🔺🔺 END REFACTORED 🔺🔺🔺

            # --- COMMON: Loss Backward and Step ---
            if loss is not None and not torch.isnan(loss):
                # --- 🔻 MODIFIED: Use GradScaler 🔻 ---
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # Unscale gradients for clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                # --- 🔺 END MODIFIED 🔺 ---

                total_loss += loss.item()

            progress_bar_postfix = {"loss": f"{loss.item():.4f}" if loss else "N/A", "task": progress_bar_task}
            if num_experts > 1:
                progress_bar_postfix["expert"] = expert_to_train_id
            progress_bar.set_postfix(progress_bar_postfix)
            # --- END OF LOOP ---

        avg_loss = total_loss / config['episodes_per_epoch'] if config['episodes_per_epoch'] > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1} finished. Average Loss: {avg_loss:.4f} | Current LR: {current_lr:.6f}")

        # Evaluate embedding quality on the first expert
        evaluate_embedding_quality(model.experts[0], loader)

        scheduler.step()

        # --- 🔻 MODIFIED: Use checkpoint_dir argument 🔻 ---
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model checkpoint saved to {checkpoint_path}")

        # --- 🔻 NEW: Stop-after-epoch logic 🔻 ---
        if stop_after_epoch is not None and (epoch + 1) == stop_after_epoch:
            print(f"\n--- 🛑 Stopping training after epoch {epoch + 1} as requested. ---")
            break
        # --- 🔺 END NEW 🔺 ---
        # --- 🔺 END MODIFIED 🔺 ---

    print("✅ Meta-training complete.")
