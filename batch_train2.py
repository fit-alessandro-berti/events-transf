"""
batch_train2.py

Runs main.py with multiple training configurations in parallel threads.
Each job trains a model with --epochs 50, --episodic_label_shuffle yes,
and --training_strategy episodic.

This version:
- Base / default = MoE with 8 experts (instead of 4)
- Explicitly runs MoE=2 and MoE=8
- All jobs support --resume (just add --resume when launching if needed)
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

# Base configuration (defaults from config.py, but we override MoE here)
BASE_CONFIG = {
    "num_experts": 8,           # ← CHANGED: now 8 experts by default
    "embedding_strategy": "learned",
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
}

# --- Job Definitions ---
TRAINING_JOBS = []

# 1. MoE = 2 experts
job_moe_2 = BASE_CONFIG.copy()
job_moe_2["name"] = "chk_episodic_moe_2"
job_moe_2["num_experts"] = 2
TRAINING_JOBS.append(job_moe_2)

# 2. MoE = 8 experts (this is now the "default" / base run)
job_moe_8 = BASE_CONFIG.copy()
job_moe_8["name"] = "chk_episodic_moe_8_default"
# num_experts already 8 → no change needed
TRAINING_JOBS.append(job_moe_8)

# (You can add more variations here if you want, e.g. different d_model, etc.)

# --- End Configuration ---


def run_training_job(job_config: dict):
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
        "--cleanup_checkpoints",                     # optional: saves disk space
        # ← Add "--resume" here if you want to resume this specific job
        # "--resume",
    ]

    # Add all dynamic parameters
    for key, value in job_config.items():
        if key == "name":
            continue
        cmd.append(f"--{key}")
        cmd.append(str(value))

    cmd_str = ' '.join(cmd)
    print(f"[Thread: {job_name}] 🚀 STARTING")
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

        if result.returncode == 0:
            print(f"[Thread: {job_name}] ✅ FINISHED (Success)")
        else:
            print(f"[Thread: {job_name}] ❌ FINISHED (Error: {result.returncode})")

    except Exception as e:
        error_msg = f"--- LAUNCH FAILED ---\nFailed to start {job_name}: {e}\nCommand: {cmd_str}\n"
        print(f"[Thread: {job_name}] 💥 CRITICAL FAILURE: {e}")
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(error_msg)


def main():
    print("--- 🏁 Starting Batch Training (MoE=2 and MoE=8) ---")

    if not TRAINING_SCRIPT.exists():
        print(f"❌ ERROR: main.py not found at {TRAINING_SCRIPT}")
        sys.exit(1)

    LOG_OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Logs will be saved to: {LOG_OUTPUT_DIR.resolve()}")
    print(f"Checkpoints will be in: {CHECKPOINTS_BASE_DIR.resolve()}")
    print(f"Total jobs to run: {len(TRAINING_JOBS)}")
    print("-" * 30 + "\n")

    threads = []
    for job in TRAINING_JOBS:
        t = threading.Thread(target=run_training_job, args=(job,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n" + "-" * 30)
    print("🎉 All training jobs are complete.")
    print("To resume a specific job, just re-run this script and add `--resume` manually,")
    print("or launch main.py directly with `--resume --checkpoint_dir checkpoints/<folder>`")


if __name__ == "__main__":
    main()
