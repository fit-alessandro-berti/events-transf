#!/usr/bin/env python3
"""
batch_train2.py

Runs main.py with multiple training configurations in parallel threads.
Default now uses 8 experts (MoE=8) instead of 4.
Added --resume support for all jobs.
"""

import sys
import subprocess
import threading
from pathlib import Path

# --- Configuration ---

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_SCRIPT = PROJECT_ROOT / "main.py"
CHECKPOINTS_BASE_DIR = PROJECT_ROOT / "checkpoints"
LOG_OUTPUT_DIR = PROJECT_ROOT / "training_output"

# --- NEW: Default base configuration (MoE = 8) ---
BASE_CONFIG = {
    "num_experts": 8,           # ← CHANGED from 4 to 8
    "embedding_strategy": "learned",
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
}

# --- TRAINING JOBS (each changes ONE parameter from the new base) ---
TRAINING_JOBS = []

# 1. MoE = 4 (for comparison)
job_moe_4 = BASE_CONFIG.copy()
job_moe_4["name"] = "chk_episodic_moe_4"
job_moe_4["num_experts"] = 4
TRAINING_JOBS.append(job_moe_4)

# 2. MoE = 8 (this is now the default/base run)
job_moe_8 = BASE_CONFIG.copy()
job_moe_8["name"] = "chk_episodic_moe_8"
job_moe_8["num_experts"] = 8
TRAINING_JOBS.append(job_moe_8)

# 3. MoE = 16 (ablation)
job_moe_16 = BASE_CONFIG.copy()
job_moe_16["name"] = "chk_episodic_moe_16"
job_moe_16["num_experts"] = 16
TRAINING_JOBS.append(job_moe_16)

# 4. pretrained embeddings instead of learned
job_pretrained = BASE_CONFIG.copy()
job_pretrained["name"] = "chk_episodic_pretrained"
job_pretrained["embedding_strategy"] = "pretrained"
TRAINING_JOBS.append(job_pretrained)

# 5–9. Architecture ablations (same as before)
for d_model, name in [(128, "d128"), (512, "d512")]:
    job = BASE_CONFIG.copy()
    job["name"] = f"chk_episodic_{name}"
    job["d_model"] = d_model
    TRAINING_JOBS.append(job)

for n_heads, name in [(4, "h4"), (16, "h16")]:
    job = BASE_CONFIG.copy()
    job["name"] = f"chk_episodic_{name}"
    job["n_heads"] = n_heads
    TRAINING_JOBS.append(job)

for n_layers, name in [(4, "l4"), (8, "l8")]:
    job = BASE_CONFIG.copy()
    job["name"] = f"chk_episodic_{name}"
    job["n_layers"] = n_layers
    TRAINING_JOBS.append(job)


# --- End Configuration ---


def run_training_job(job_config: dict, resume: bool = False):
    job_name = job_config["name"]
    checkpoint_dir = CHECKPOINTS_BASE_DIR / job_name
    log_file_path = LOG_OUTPUT_DIR / f"{job_name}.txt"

    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    cmd = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--epochs", "50",
        "--episodic_label_shuffle", "yes",
        "--training_strategy", "episodic",
        "--checkpoint_dir", str(checkpoint_dir),
        "--cleanup_checkpoints",                     # saves disk space
    ]

    # Add --resume flag if requested and a checkpoint already exists
    if resume and any((checkpoint_dir / f).exists() for f in os.listdir(checkpoint_dir) if f.startswith("model_epoch_")):
        cmd.append("--resume")
        print(f"[Thread: {job_name}] Resuming from existing checkpoint")

    # Add all dynamic parameters
    for key, value in job_config.items():
        if key == "name":
            continue
        cmd.append(f"--{key}")
        cmd.append(str(value))

    cmd_str = " ".join(cmd)
    print(f"[Thread: {job_name}] {'RESUMING' if resume else 'STARTING'}")
    print(f"[Thread: {job_name}]   > Log: {log_file_path.name}")
    print(f"[Thread: {job_name}]   > Cmd: {cmd_str}\n")

    try:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )

        status = "Success" if result.returncode == 0 else f"Error {result.returncode}"
        print(f"[Thread: {job_name}] ✅ FINISHED ({status})")

    except Exception as e:
        error_msg = f"--- LAUNCH FAILED ---\nFailed to start {job_name}: {e}\nCommand: {cmd_str}\n"
        print(f"[Thread: {job_name}] 💥 CRITICAL FAILURE: {e}")
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(error_msg)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch training with MoE=8 as new default + optional resume")
    parser.add_argument("--resume", action="store_true", help="Resume all jobs that already have checkpoints")
    args = parser.parse_args()

    print("--- 🏁 Starting Batch Training (MoE=8 default) ---")
    print(f"Resume mode: {'ON' if args.resume else 'OFF'}")

    if not TRAINING_SCRIPT.exists():
        print(f"❌ ERROR: main.py not found at {TRAINING_SCRIPT}")
        sys.exit(1)

    LOG_OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Logs → {LOG_OUTPUT_DIR.resolve()}")
    print(f"Checkpoints → {CHECKPOINTS_BASE_DIR.resolve()}")
    print(f"Total jobs: {len(TRAINING_JOBS)}\n")

    threads = []
    for job in TRAINING_JOBS:
        t = threading.Thread(target=run_training_job, args=(job, args.resume))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n" + "-" * 30)
    print("🎉 All training jobs completed.")


if __name__ == "__main__":
    main()
