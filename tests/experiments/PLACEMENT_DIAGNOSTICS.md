# Placement cost experiments

Branch: `615.71.09/experiment/placement-diagnostics`.

This branch adds optional measurements to the block-coverage/bounded-refill
experiment. It does not change admission, copy masks, reservations, VAB ordering,
eviction decisions, thresholds or retry logic. Diagnostics default off.

## Build, verify and run

Rebuild the Linux module and reload with your current settings plus:

```bash
uvm_perf_placement_diagnostics_enable=1
```

Keep threshold 1024, pressure 85%, base observations 2, coverage enabled at 25%
with sparse observations 4, indicated-pages disabled, bounded replacement enabled,
reserve 64 MiB and limit 32 MiB. Keep extra eviction probation, refault, sparse
prefetch, backoff, recovery and region migration disabled, as in the current run.
Retain eager batch 16 and your existing VAB settings and verified physical base.

Before benchmarking, verify the exact binary path that you pass to `insmod`:

```bash
modinfo -p ../open-gpu-kernel-modules/kernel-open/nvidia-uvm.ko \
  | grep placement_diagnostics
cat /sys/module/nvidia_uvm/parameters/uvm_perf_placement_diagnostics_enable
grep . /sys/module/nvidia_uvm/parameters/uvm_perf_diag_*
```

The loaded switch must read `Y` and the 17 diagnostic counters must exist, even
before they increase. A Git commit in monitor metadata cannot verify the loaded
kernel binary. The parameters are read-only after loading.

From the vLLM fork, use the existing monitor wrapper:

```bash
BATCH_SIZES="32" RESULT_ROOT=results/placement-diag-on \
  bash ../../open-gpu-kernel-modules/tests/experiments/run_vllm_monitored.sh
```

Use separate output roots for these comparisons and reload between settings:

| Run | Diagnostics | Coverage admission | Purpose |
|---|---:|---:|---|
| A | 0 | 1 | Current throughput with timing disabled |
| B | 1 | 1 | All requested stage timings and outcomes |
| C | 1 | 0 | Compare costs of coverage admission on the same binary |
| D | 1 | 1 | Repeat B at batch 8 to check low-pressure behavior |

Alternate and repeat A/B to estimate observation overhead before interpreting
small throughput differences. Then alternate B/C with identical host conditions,
external VRAM reservation and workload. D uses `BATCH_SIZES="8"`. Do not alter
hardware threshold or copy size during these comparisons. No profiler or PCIe
bandwidth sampling is required for these kernel counters.

## Timing boundaries

Each stage exposes cumulative `uvm_perf_diag_<stage>_calls` and
`uvm_perf_diag_<stage>_ns` through sysfs. Counts include failed/empty calls where
the instrumented scope executes.

| Stage | Measured scope |
|---|---|
| `counter_service` | A fetched notification service pass, including preprocessing, VA-space locks and deferred optional population. Excludes buffer fetching. |
| `counter_block` | One logical-block notification group: evidence expansion, block locks, admission, native service and notification flags/clear submission. Ends before optional queue insertion/flush. |
| `population_submit` | One optional `prefetch_block()` call, including block locks, mask preparation, make-resident, mapping and tracker attachment. Empty or failed calls count. |
| `population_wait` | The final tracker wait/deinitialization for a capacity-admitted population attempt. Includes empty work and context-allocation failure after admission. Denied reservations do not count. |
| `eviction_select` | One background root-selection attempt, including VAB snapshot, PMM lock acquisition and root scan. Invalid snapshots and exhausted scans count. |
| `eviction_submit` | Background `evict_root_chunk()` call, including VA-block migration submission and any synchronous waits/error cleanup inside that call. Excludes native mandatory/PMA eviction calls. |
| `eviction_drain` | Background `free_root_chunk()` for each successfully submitted root, including its tracker wait and release. Cleanup inside a failed submission belongs to submit. |

These are elapsed wall times around CPU-side functions, including scheduling and
lock waits. They do not measure CPU execution cycles or GPU engine busy time.
In particular, submit time can include internal waits and is not pure enqueue
latency. Block processing can include native GPU servicing. `counter_block`
excludes deferred optional population, whereas `counter_service` includes it.

The inclusive service timer nests other stages. Calls across workers and GPUs
can also overlap, so **do not add stage totals or interpret them as a serial
runtime decomposition**. A small final wait does not prove copies were cheap:
asynchronous work may already have completed while other work ran. These counters
do not directly measure remote GPU loads or establish PCIe saturation.

## Empty and nonempty populations

Three cumulative counters classify completed capacity-admitted optional attempts:

- `uvm_perf_diag_admitted_empty`: zero accounted changed-residency bytes.
- `uvm_perf_diag_admitted_nonempty`: positive accounted changed-residency bytes.
- `uvm_perf_diag_admitted_failed`: failure, including capacity stop, context
  allocation failure, submission/mapping failure or final tracker-wait failure.

Empty/nonempty are mutually exclusive. Failed is an overlapping subset, and a
failed attempt can already have moved data. A zero accounted byte result does
not prove no internal hardware work occurred. Counters describe an entire
reservation, which can include multiple block calls below pressure. They are
different from per-block submission counts and admission grants before capacity
reservation. Early one-shot skips and denied reservations are outside this
outcome classification; existing counters retain their meanings.

## Monitor output and interpretation

The updated monitor records raw values in snapshots and samples, before/after
deltas and rates in changes files, and a compact report in:

```text
summary.json -> derived -> placement_diagnostics
```

The report includes enabled/disabled/unsupported status, each stage's calls,
total seconds and mean microseconds, plus empty/nonempty/failed counts and the
empty fraction. It uses a common before/after interval. Missing or reset counter
coverage remains unknown; no calls produce a null mean. Counter pairs are read
separately, so active snapshots can be slightly skewed. Full-run totals include
model loading, cold and warm passes and teardown. Use sample endpoint deltas for
separate warm intervals rather than assigning full-run timing to decode alone.

Useful next decisions based on the measurements:

- High empty fraction with material submit time supports avoiding redundant
  population preparation; fewer admission grants alone does not establish this.
- High block-processing cost supports investigating evidence/clear/lock handling.
- High selection cost supports investigating scan efficiency and PMM lock waits.
- High submit/drain/wait times support deeper migration and synchronization
  profiling, without assuming every measured second lies on the inference path.

All counters aggregate the loaded driver across GPUs and workloads. Isolate the
test workload. Enabled hooks perform timestamp reads and atomic updates and may
perturb timing. Disabled timing hooks read the load-time flag but perform no
timestamp reads or counter updates. Outcome hooks also skip updates when off.
Local tests exercise production routines with deterministic clocks and mocked
hardware. Linux compilation and GPU performance/overhead validation remain
required on the target machine.
