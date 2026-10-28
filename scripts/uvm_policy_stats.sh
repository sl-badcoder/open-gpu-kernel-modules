#!/usr/bin/env bash

set -u

readonly PARAM_DIR="${UVM_PARAM_DIR:-/sys/module/nvidia_uvm/parameters}"
readonly PREFIX="uvm_cpu_preferred_"

readonly METRICS=(
    access_counter_batches
    access_counter_notifications
    access_counter_value
    block_promotions
    promoted_bytes
    range_prefetches
    prefetch_retries
    prefetched_bytes
    prefetch_capacity_stops
    demotion_transfers
    demoted_bytes
    eager_eviction_attempts
    eager_eviction_successes
    eager_eviction_failures
    eager_reclaimed_bytes
)

usage()
{
    cat <<EOF
Usage:
  $0 show
  $0 measure -- command [args...]

"show" prints the current cumulative UVM policy counters.
"measure" runs a command and prints counter deltas plus elapsed wall time.

Notes:
  Counters are driver-global, so keep unrelated GPU work idle while measuring.
  access_counter_value is sampled hardware activity, not an exact transaction
  count; hardware counter values can saturate before UVM fetches them.
EOF
}

check_counters()
{
    local metric

    for metric in "${METRICS[@]}"; do
        if [[ ! -r "${PARAM_DIR}/${PREFIX}${metric}" ]]; then
            echo "Missing ${PARAM_DIR}/${PREFIX}${metric}" >&2
            echo "Build, install, and load the updated nvidia_uvm module first." >&2
            exit 1
        fi
    done
}

read_counter()
{
    local metric="$1"
    local value

    IFS= read -r value < "${PARAM_DIR}/${PREFIX}${metric}"
    printf '%s' "$value"
}

is_byte_metric()
{
    case "$1" in
        promoted_bytes|prefetched_bytes|demoted_bytes|eager_reclaimed_bytes)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

print_value()
{
    local metric="$1"
    local value="$2"

    if is_byte_metric "$metric"; then
        awk -v value="$value" 'BEGIN { printf "%-16s (%9.2f MiB)", value, value / 1048576 }'
    else
        printf '%-16s' "$value"
    fi
}

show()
{
    local metric
    local value

    printf '%-38s %s\n' "metric" "value"
    printf '%-38s %s\n' "------" "-----"
    for metric in "${METRICS[@]}"; do
        value="$(read_counter "$metric")"
        printf '%-38s ' "$metric"
        print_value "$metric" "$value"
        printf '\n'
    done
}

measure()
{
    local metric
    local status
    local start_ns
    local end_ns
    local elapsed_ns
    local delta
    declare -A before
    declare -A after

    if [[ "${1:-}" == "--" ]]; then
        shift
    fi
    if [[ "$#" -eq 0 ]]; then
        usage >&2
        exit 2
    fi

    for metric in "${METRICS[@]}"; do
        before["$metric"]="$(read_counter "$metric")"
    done

    start_ns="$(date +%s%N)"
    "$@"
    status=$?
    end_ns="$(date +%s%N)"

    for metric in "${METRICS[@]}"; do
        after["$metric"]="$(read_counter "$metric")"
    done

    elapsed_ns=$((end_ns - start_ns))
    printf '\ncommand_exit_status: %d\n' "$status"
    awk -v elapsed_ns="$elapsed_ns" 'BEGIN { printf "elapsed_seconds: %.6f\n", elapsed_ns / 1000000000 }'
    printf '%-38s %s\n' "metric_delta" "value"
    printf '%-38s %s\n' "------------" "-----"

    for metric in "${METRICS[@]}"; do
        delta=$((after[$metric] - before[$metric]))
        printf '%-38s ' "$metric"
        print_value "$metric" "$delta"
        printf '\n'
    done

    return "$status"
}

case "${1:-show}" in
    show)
        check_counters
        show
        ;;
    measure)
        shift
        check_counters
        measure "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
