#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: ./run_example1_batch.sh NUM_LOGS OUTPUT_DIR [TRACES_PER_LOG] [CASE_ARRIVAL] [BASE_TRACES] [SEED_BASE]

Runs example1.py NUM_LOGS times, each as a separate background process, and writes
logs to OUTPUT_DIR/log_XXXX.xes (zero-padded).

Environment:
  MAX_PARALLEL   Optional cap on concurrent background jobs (0 = no cap).

Examples:
  ./run_example1_batch.sh 1000 ./out 1000 3600 50 17
  MAX_PARALLEL=8 ./run_example1_batch.sh 200 ./out 500 3600 50 100
USAGE
  exit 0
fi

num_logs="${1:-10}"
out_dir="${2:-./out}"
traces_per_log="${3:-1000}"
case_arrival="${4:-3600}"
base_traces="${5:-50}"
seed_base="${6:-17}"
max_parallel="${MAX_PARALLEL:-0}"

mkdir -p "$out_dir"

width=${#num_logs}
pids=()

for i in $(seq 1 "$num_logs"); do
  printf -v file_idx "%0${width}d" "$i"
  out_file="${out_dir}/log_${file_idx}.xes"
  seed=$((seed_base + i - 1))
  python perform_simulation.py \
    --num-logs 1 \
    --traces-per-log "$traces_per_log" \
    --case-arrival-ratio "$case_arrival" \
    --base-traces "$base_traces" \
    --seed "$seed" \
    --output "$out_file" &
  pids+=("$!")

  if [[ "$max_parallel" -gt 0 ]]; then
    while [[ "$(jobs -rp | wc -l)" -ge "$max_parallel" ]]; do
      sleep 0.5
    done
  fi
done

wait
echo "Completed ${num_logs} log(s) in ${out_dir}"
