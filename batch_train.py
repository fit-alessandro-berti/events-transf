#!/usr/bin/env python3
"""
batch_train.py

Runs training.py with multiple configurations in parallel threads.
Each job's output (stdout/stderr) is redirected to a unique log file
in the ./training_output directory.
"""

import sys
import os
import subprocess
import threading
from pathlib import Path

# --- Configuration ---

# 1. Define the root of your project
# (This script assumes it's in the same directory as training.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# 2. Path to the training script
TRAINING_SCRIPT = PROJECT_ROOT / "training.py"

# 3. Directory to save checkpoint folders in
CHECKPOINTS_BASE_DIR = PROJECT_ROOT / "checkpoints"

# 4. Directory to save log files in
LOG_OUTPUT_DIR = PROJECT_ROOT / "training_output"

# 5. Define the training jobs to run
# Each job is a dictionary specifying its unique configuration.
TRAINING_JOBS = [
    {
        "name": "chk_shuffle_yes_episodic",
        "shuffle": "yes",
        "strategy": "episodic",
    },
    {
        "name": "chk_shuffle_yes_retrieval",
        "shuffle": "yes",
        "strategy": "retrieval",
    },
    {
        "name": "chk_shuffle_no_episodic",
        "name_short": "NoShuffle-Episodic",
        "shuffle": "no",
        "strategy": "episodic",
    },
    {
        "name": "chk_shuffle_no_retrieval",
        "shuffle": "no",
        "strategy": "retrieval",
    }
]


# --- End Configuration ---


def run_training_job(job_config: dict):
    """
    This function is executed by each thread.
    It builds and runs a single training.py command.
    """

    # 1. Get job details
    job_name = job_config["name"]
    shuffle = job_config["shuffle"]
    strategy = job_config["strategy"]

    # 2. Define paths for this specific job
    checkpoint_dir = CHECKPOINTS_BASE_DIR / job_name
    log_file_path = LOG_OUTPUT_DIR / f"{job_name}.txt"

    # 3. Create the directories
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # 4. Build the subprocess command
    #    We use sys.executable to ensure we use the same Python interpreter.
    cmd = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--epochs", "50",
        "--episodic_label_shuffle", shuffle,
        "--training_strategy", strategy,
        "--checkpoint_dir", str(checkpoint_dir),
        "--cleanup_checkpoints"  # Saves disk space by only keeping the final model
    ]

    cmd_str = ' '.join(cmd)
    print(f"[Thread: {job_name}] 🚀 STARTING")
    print(f"[Thread: {job_name}]   > Log: {log_file_path.name}")
    print(f"[Thread: {job_name}]   > Cmd: {cmd_str}\n")

    # 5. Run the command and redirect output
    try:
        # Open the log file in 'w' mode (overwrites old logs)
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            # We use subprocess.run, which is the modern equivalent
            # of os.system but with better I/O control.
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                check=False  # Don't raise an error if training.py fails
            )

        if result.returncode == 0:
            print(f"[Thread: {job_name}] ✅ FINISHED (Success)")
        else:
            print(f"[Thread: {job_name}] ❌ FINISHED (Error: {result.returncode})")

    except Exception as e:
        # Catch errors if the script *itself* fails to launch
        error_msg = f"--- LAUNCH FAILED ---\nFailed to start {job_name}: {e}\nCommand: {cmd_str}\n"
        print(f"[Thread: {job_name}] 💥 CRITICAL FAILURE: {e}")
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(error_msg)


def main():
    """
    Main function to set up and launch all training threads.
    """

    print("--- 🏁 Starting Batch Training ---")

    # --- 1. Initial Checks ---
    if not TRAINING_SCRIPT.exists():
        print(f"❌ ERROR: training.py not found at {TRAINING_SCRIPT}")
        print("Please make sure this script is in the same directory.")
        sys.exit(1)

    # Create the main output/log directory
    LOG_OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Logs will be saved to: {LOG_OUTPUT_DIR.resolve()}")
    print(f"Checkpoints will be in: {CHECKPOINTS_BASE_DIR.resolve()}")
    print("-" * 30 + "\n")

    # --- 2. Launch Threads ---
    threads = []
    for job in TRAINING_JOBS:
        # Create a thread for each job, targeting the run_training_job function
        t = threading.Thread(target=run_training_job, args=(job,))
        threads.append(t)
        t.start()  # Start the thread (and the job)

    # --- 3. Wait for all threads to complete ---
    for t in threads:
        t.join()  # This blocks the main script until the thread is done

    print("\n" + "-" * 30)
    print("🎉 All training jobs are complete.")
    print("Check log files for details.")


if __name__ == "__main__":
    main()
