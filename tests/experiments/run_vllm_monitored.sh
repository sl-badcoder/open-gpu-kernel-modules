#!/usr/bin/env bash
set -euo pipefail

# Run from the vLLM fork directory containing benchmarks/benchmark_uvm.py.
# Override MONITOR if this script was copied away from monitor_uvm.py.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
monitor=${MONITOR:-"$script_dir/monitor_uvm.py"}
gpu=${GPU_ID:-0}
interval=${SAMPLE_INTERVAL:-2}
result_root=${RESULT_ROOT:-results/ember-adaptive}
run_id=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$result_root"

for batch in ${BATCH_SIZES:-24}; do
  run_dir="$result_root/kv-adp3-batch-${batch}-${run_id}-$$"
  python3 "$monitor" --gpu "$gpu" --interval "$interval" \
    --output-dir "$run_dir" -- \
    .venv/bin/python benchmarks/benchmark_uvm.py \
      --model Qwen/Qwen2.5-3B-Instruct \
      --allocator managed --kv-cache-gib 12 \
      --require-oversubscription \
      --num-prompts 32 --batch-size "$batch" \
      --input-tokens 10240 --output-tokens 256 \
      --repeats 3 \
      --output "$run_dir/benchmark.json"
done
