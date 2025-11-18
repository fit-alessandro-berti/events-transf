# File: batch_test_logs.py
#!/usr/bin/env python3
"""
batch_test_logs.py

... (docstring unchanged) ...
"""

import os
import re
import argparse
import subprocess
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# Helper: strip everything that is not alphanumeric or underscore
# ----------------------------------------------------------------------
def alphanumeric(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "", s.replace(" ", "_"))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch-run testing.py on all relevant logs")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Checkpoint directory (used for model loading AND for output filename)")
    parser.add_argument("--checkpoint_epoch", type=int, default=None,
                        help="Specific epoch to test (default: latest)")
    parser.add_argument("--test_log_name", type=str, default=None,
                        help="Name of the test log as defined in config.py['log_paths']['testing'] (can be overridden by --test_mode)")
    parser.add_argument("--test_mode", type=str, default=None,
                        choices=["meta_learning", "retrieval_augmented"],
                        help="Test mode (default: whatever is currently in CONFIG)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent
    logs_dir = project_root / "logs"
    testing_script = project_root / "testing.py"
    output_dir = project_root / "testing_output"
    output_dir.mkdir(exist_ok=True)

    if not testing_script.exists():
        print(f"ERROR: testing.py not found at {testing_script}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Find all logs that start with at least four zeros
    # ------------------------------------------------------------------
    pattern = re.compile(r"^0{4,}.*\.xes\.gz$", re.IGNORECASE)
    target_logs = [p for p in logs_dir.iterdir() if p.is_file() and pattern.match(p.name)]

    if not target_logs:
        print("No logs found that start with at least four zeros.")
        return

    print(f"Found {len(target_logs)} logs to test:")
    for p in target_logs:
        print(f"  - {p.name}")
    print()

    # ------------------------------------------------------------------
    # Build the base command (everything that is common)
    # ------------------------------------------------------------------
    cmd_base = [
        sys.executable, str(testing_script),
        "--checkpoint_dir", str(args.checkpoint_dir),
    ]

    if args.checkpoint_epoch is not None:
        cmd_base += ["--checkpoint_epoch", str(args.checkpoint_epoch)]

    # This argument will be overridden in the loop, but if the loop
    # logic changes, it's good to have a default.
    if args.test_log_name:
        cmd_base += ["--test_log_name", args.test_log_name]

    if args.test_mode:
        cmd_base += ["--test_mode", args.test_mode]

    # ------------------------------------------------------------------
    # Prepare filename components (once, they are the same for every log)
    # ------------------------------------------------------------------
    chk_name = Path(args.checkpoint_dir).name
    chk_part = alphanumeric(chk_name)

    epoch_part = f"{args.checkpoint_epoch}" if args.checkpoint_epoch is not None else "latest"

    mode_part = alphanumeric(args.test_mode) if args.test_mode else "defaultmode"

    # ------------------------------------------------------------------
    # Run testing.py for each log
    # ------------------------------------------------------------------
    for log_path in target_logs:
        # The actual log name without path/extension (used only for the output filename)
        log_stem = re.sub(r'\.xes(\.gz)?$', '', log_path.name, flags=re.IGNORECASE)
        log_part = alphanumeric(log_stem)

        # Build the final output filename
        out_filename = f"{chk_part}_{log_part}_{mode_part}_{epoch_part}.txt"
        out_path = output_dir / out_filename

        print(f"Testing {log_path.name} → {out_path.name}")

        # --- 🔻 MODIFIED: Pass the full path to testing.py 🔻 ---
        # We pass the full path string. testing.py now accepts this.
        # This replaces any default --test_log_name from cmd_base.
        cmd = cmd_base + ["--test_log_name", str(log_path)]
        # --- 🔺 END MODIFIED 🔺 ---

        # Run and capture everything
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # merge stderr into stdout
                text=True,
                check=False                 # we want to continue even if testing.py returns non-zero
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"COMMAND: {' '.join(cmd)}\n")
                f.write(f"RETURN CODE: {result.returncode}\n")
                f.write("=" * 80 + "\n")
                f.write(result.stdout)
            print(f"   → Finished (return code {result.returncode})")
        except Exception as e:
            error_msg = f"FAILED to run testing.py for {log_path.name}: {e}\n"
            print(error_msg)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(error_msg)

    print("\nAll done! Outputs are in:", output_dir)


if __name__ == "__main__":
    main()
