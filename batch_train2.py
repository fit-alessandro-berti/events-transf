#!/usr/bin/env python3
"""
batch_train2.py

Runs main.py with multiple training configurations in parallel threads.
Each job trains a model with --epochs 50, --episodic_label_shuffle yes,
and --training_strategy episodic.

Each job changes exactly one parameter from the default configuration.
"""

import sys
import os
import subprocess
import threading
from pathlib import Path

# --- Configuration ---

# 1. Define the root of your project
PROJECT_ROOT = Path(__file__).resolve().parent

# 2. Path to the main training script to execute
TRAINING_SCRIPT = PROJECT_ROOT / "main.py"

# 3. Directory to save checkpoint folders in
CHECKPOINTS_BASE_DIR = PROJECT_ROOT / "checkpoints"

# 4. Directory to save log files in
LOG_OUTPUT_DIR = PROJECT_ROOT / "training_output"

# 5. Define the "default" parameters for this batch
#    (Based on config.py defaults)
BASE_CONFIG = {
    "num_experts": 4,
    "embedding_strategy": "learned",
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
}

# 6. Define the training jobs to run
#    Each job is a copy of BASE_CONFIG with one parameter changed.
TRAINING_JOBS = []

# --- Job Definitions ---

# 1. num_experts = 2 (Default is 4)
job_moe_2 = BASE_CONFIG.copy()
job_moe_2["name"] = "chk_episodic_moe_2"
job_moe_2["num_experts"] = 2
TRAINING_JOBS.append(job_moe_2)

# 2. num_experts = 4 (This is the default base run)
job_moe_4 = BASE_CONFIG.copy()
job_moe_4["name"] = "chk_episodic_moe_4_default"
# No parameter change needed, it's already the default
TRAINING_JOBS.append(job_moe_4)

# 3. embedding_strategy = pretrained (Default is 'learned')
job_pretrained = BASE_CONFIG.copy()
job_pretrained["name"] = "chk_episodic_pretrained"
job_pretrained["embedding_strategy"] = "pretrained"
TRAINING_JOBS.append(job_pretrained)

# 4. d_model = 128 (Default is 256)
job_d128 = BASE_CONFIG.copy()
job_d128["name"] = "chk_episodic_d128"
job_d128["d_model"] = 128
TRAINING_JOBS.append(job_d128)

# 5. d_model = 512 (Default is 256)
job_d512 = BASE_CONFIG.copy()
job_d512["name"] = "chk_episodic_d512"
job_d512["d_model"] = 512
TRAINING_JOBS.append(job_d512)

# 6. n_heads = 4 (Default is 8)
job_h4 = BASE_CONFIG.copy()
job_h4["name"] = "chk_episodic_h4"
job_h4["n_heads"] = 4
TRAINING_JOBS.append(job_h4)

# 7. n_heads = 16 (Default is 8)
job_h16 = BASE_CONFIG.copy()
job_h16["name"] = "chk_episodic_h16"
job_h16["n_heads"] = 16
TRAINING_JOBS.append(job_h16)

# 8. n_layers = 4 (Default is 6)
job_l4 = BASE_CONFIG.copy()
job_l4["name"] = "chk_episodic_l4"
job_l4["n_layers"] = 4
TRAINING_JOBS.append(job_l4)

# 9. n_layers = 8 (Default is 6)
job_l8 = BASE_CONFIG.copy()
job_l8["name"] = "chk_episodic_l8"
job_l8["n_layers"] = 8
TRAINING_JOBS.append(job_l8)


# --- End Configuration ---


def run_training_job(job_config: dict):
    """
    This function is executed by each thread.
    It builds and runs a single main.py command based on the job_config.
    """

    # 1. Get job name and define paths
    job_name = job_config["name"]
    checkpoint_dir = CHECKPOINTS_BASE_DIR / job_name
    log_file_path = LOG_OUTPUT_DIR / f"{job_name}.txt"

    # 2. Create the directories
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # 3. Build the subprocess command
    # Start with the static, common parameters
    cmd = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--epochs", "50",
        "--episodic_label_shuffle", "yes",
        "--training_strategy", "episodic",
        "--checkpoint_dir", str(checkpoint_dir),
        # Add cleanup flag to save disk space on batch runs
        "--cleanup_checkpoints",
    ]

    # 4. Add all dynamic parameters from the job config
    for key, value in job_config.items():
        if key == "name":
            continue  # 'name' is for the folder, not an arg

        # Convert python key to command line argument
        # e.g., 'num_experts' -> '--num_experts'
        arg_name = f"--{key}"
        cmd.append(arg_name)
        cmd.append(str(value))  # Ensure value is a string for subprocess

    cmd_str = ' '.join(cmd)
    print(f"[Thread: {job_name}] 🚀 STARTING")
    print(f"[Thread: {job_name}]   > Log: {log_file_path.name}")
    print(f"[Thread: {job_name}]   > Cmd: {cmd_str}\n")

    # 5. Run the command and redirect output
    try:
        # Open the log file in 'w' mode (overwrites old logs)
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
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
    """
    Main function to set up and launch all training threads.
    """

    print("--- 🏁 Starting Batch Training (Script 2) ---")

    # --- 1. Initial Checks ---
    if not TRAINING_SCRIPT.exists():
        print(f"❌ ERROR: main.py not found at {TRAINING_SCRIPT}")
        print("Please make sure this script is in the same directory.")
        sys.exit(1)

    # Create the main output/log directory
    LOG_OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Logs will be saved to: {LOG_OUTPUT_DIR.resolve()}")
    print(f"Checkpoints will be in: {CHECKPOINTS_BASE_DIR.resolve()}")
    print(f"Total jobs to run: {len(TRAINING_JOBS)}")
    print("-" * 30 + "\n")

    # --- 2. Launch Threads ---
    threads = []
    for job in TRAINING_JOBS:
        t = threading.Thread(target=run_training_job, args=(job,))
        threads.append(t)
        t.start()

    # --- 3. Wait for all threads to complete ---
    for t in threads:
        t.join()

    print("\n" + "-" * 30)
    print("🎉 All training jobs are complete.")
    print("Check log files for details.")


if __name__ == "__main__":
    main()
