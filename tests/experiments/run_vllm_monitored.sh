#!/usr/bin/env bash
set -euo pipefail

# Run from the vLLM fork directory containing benchmarks/benchmark_uvm.py.
# Override MONITOR if this script was copied away from monitor_uvm.py.
# To sweep thresholds, set ACCESS_COUNTER_THRESHOLDS="1024 10024 32768 65535"
# and RELOAD_SCRIPT=/path/to/your/reload.sh. The reload script must accept
# uvm_perf_access_counter_threshold=N and handle sudo itself.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
monitor=${MONITOR:-"$script_dir/monitor_uvm.py"}
gpu=${GPU_ID:-0}
interval=${SAMPLE_INTERVAL:-2}
result_root=${RESULT_ROOT:-results/ember-adaptive}
run_id=$(date -u +%Y%m%dT%H%M%SZ)
thresholds=()
if [[ -n ${ACCESS_COUNTER_THRESHOLDS:-} ]]; then
  read -r -a thresholds <<< "$ACCESS_COUNTER_THRESHOLDS"
  if [[ ${#thresholds[@]} -eq 0 ]]; then
    echo "ACCESS_COUNTER_THRESHOLDS must contain at least one threshold" >&2
    exit 2
  fi
  if [[ -z ${RELOAD_SCRIPT:-} || ! -f $RELOAD_SCRIPT || ! -r $RELOAD_SCRIPT ]]; then
    echo "Set RELOAD_SCRIPT to a readable reload script before sweeping thresholds" >&2
    exit 2
  fi
  # Validate the entire sweep before unloading a module or starting a benchmark.
  for index in "${!thresholds[@]}"; do
    value=${thresholds[$index]}
    if [[ ! $value =~ ^[0-9]{1,5}$ ]]; then
      echo "Invalid access-counter threshold: $value (expected 1..65535)" >&2
      exit 2
    fi
    value=$((10#$value))
    if (( value < 1 || value > 65535 )); then
      echo "Invalid access-counter threshold: $value (expected 1..65535)" >&2
      exit 2
    fi
    thresholds[$index]=$value
  done
else
  # Preserve the original behavior: benchmark the currently loaded module.
  thresholds=(current)
fi

# The optional wrapper must exec its foreground reservation process so $! owns
# the GPU client itself. Never kill unrelated/manual reservation processes.
reservation_pid=""
reservation_command=()
reservation_startup_seconds=${RESERVATION_STARTUP_SECONDS:-3}
if [[ -n ${RESERVATION_GIB:-} ]]; then
  if [[ -n ${RESERVATION_SCRIPT:-} ]]; then
    echo "Use RESERVATION_GIB or RESERVATION_SCRIPT, not both" >&2
    exit 2
  fi
  if [[ ! $RESERVATION_GIB =~ ^[1-9][0-9]*$ ]]; then
    echo "RESERVATION_GIB must be a positive integer (passed to reserv)" >&2
    exit 2
  fi
  reservation_executable=${RESERVATION_EXECUTABLE:-./reserv}
  if [[ ! -f $reservation_executable || ! -x $reservation_executable ]]; then
    echo "RESERVATION_EXECUTABLE must name an executable reservation helper" >&2
    exit 2
  fi
  reservation_command=("$reservation_executable" "$RESERVATION_GIB")
elif [[ -n ${RESERVATION_SCRIPT:-} ]]; then
  if [[ ! -f $RESERVATION_SCRIPT || ! -r $RESERVATION_SCRIPT ]]; then
    echo "RESERVATION_SCRIPT must name a readable script" >&2
    exit 2
  fi
  reservation_command=(bash "$RESERVATION_SCRIPT")
fi
if [[ ${#reservation_command[@]} -gt 0 ]]; then
  if [[ ! $reservation_startup_seconds =~ ^[0-9]+$ ]]; then
    echo "RESERVATION_STARTUP_SECONDS must be a nonnegative integer" >&2
    exit 2
  fi
fi

stop_reservation() {
  if [[ -n $reservation_pid ]]; then
    echo "Stopping GPU reservation (PID $reservation_pid)"
    kill -TERM "$reservation_pid" 2>/dev/null || true
    # Bound cleanup even if the helper does not respond to SIGTERM.
    for ((attempt = 0; attempt < 50; ++attempt)); do
      if ! kill -0 "$reservation_pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$reservation_pid" 2>/dev/null; then
      kill -KILL "$reservation_pid" 2>/dev/null || true
    fi
    wait "$reservation_pid" 2>/dev/null || true
    reservation_pid=""
  fi
}

check_reservation() {
  if [[ ${#reservation_command[@]} -gt 0 ]] && ! kill -0 "$reservation_pid" 2>/dev/null; then
    echo "GPU reservation exited unexpectedly; check its reservation log" >&2
    exit 1
  fi
}

trap stop_reservation EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$result_root"

for threshold in "${thresholds[@]}"; do
  # A live CUDA client may prevent nvidia_uvm from unloading.
  stop_reservation
  threshold_tag=""
  if [[ $threshold != current ]]; then
    echo "Reloading module with access-counter threshold $threshold"
    bash "$RELOAD_SCRIPT" "uvm_perf_access_counter_threshold=$threshold" \
      2>&1 | tee "$result_root/reload-ac-${threshold}-${run_id}-$$.log"
    threshold_tag="-ac-${threshold}"
  fi
  if [[ ${#reservation_command[@]} -gt 0 ]]; then
    reservation_log="$result_root/reservation-ac-${threshold}-${run_id}-$$.log"
    echo "Starting GPU reservation; log: $reservation_log"
    "${reservation_command[@]}" > "$reservation_log" 2>&1 &
    reservation_pid=$!
    reservation_deadline=$((SECONDS + 10#$reservation_startup_seconds))
    while (( SECONDS < reservation_deadline )); do
      sleep 0.1
      check_reservation
    done
    check_reservation
  fi
  for batch in ${BATCH_SIZES:-24}; do
    check_reservation
    run_dir="$result_root/kv-adp3-batch-${batch}${threshold_tag}-${run_id}-$$"
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
    check_reservation
  done
done
