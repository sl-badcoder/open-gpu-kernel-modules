# Counter-indicated trigger population

Branch: `615.71.09/experiment/counter-indicated-pages`.

This experiment retains bounded refill and two-observation admission. Under
admission pressure, it limits optional trigger population to the original
access-counter notification page mask instead of filling eligible CPU-resident
pages throughout the trigger's 2 MiB VA block. Below pressure, larger trigger and
neighbor population retain their existing behavior. Native counter servicing,
counter clearing and VAB eviction selection retain their existing paths.

## Compare

Rebuild on Linux. Keep the settings in [BOUNDED_REFILL_ADMISSION.md](BOUNDED_REFILL_ADMISSION.md),
including threshold 1024, pressure 85%, observations 2, replacement enabled,
extra eviction probation disabled, full-block promotion disabled, and
refault/backoff/recovery/sparse/region migration disabled. Change only:

```bash
# Baseline, also the default:
uvm_perf_access_counter_indicated_pages_enable=0
# Experiment:
uvm_perf_access_counter_indicated_pages_enable=1
```

Reload between configurations. Run alternating baseline/experiment runs for
batch 32, then check batches 8 and 24 for regressions. Keep the monitoring output
separate, for example from the vLLM directory:

```bash
BATCH_SIZES="8 24 32" RESULT_ROOT=results/indicated-on \
  bash ../../open-gpu-kernel-modules/tests/experiments/run_vllm_monitored.sh
```

Use `results/indicated-off` for baseline. Both settings collect the new counters:

- `uvm_cpu_preferred_trigger_indicated_bytes`: successful trigger population
  bytes inside the notification mask.
- `uvm_cpu_preferred_trigger_extra_bytes`: successful trigger population bytes
  outside that mask.

The monitor captures deltas and reports `module/trigger_extra_fraction` in the
summary's derived metrics. Missing coverage or no population yields null.
Compare warm tokens/s, total trigger population bytes per output token, the
extra fraction and background allocated-root releases. In limited mode, extra
bytes should be zero for these optional trigger populations. Other native
migration paths are not attributed by these counters. Extra bytes in baseline
are speculative relative to the notification evidence, not proof of unused data.

## Implementation and limits

The notification mask is copied before native servicing can change it, then
stored by value in the bounded deferred queue. Population intersects that mask
with currently eligible CPU-resident pages and excludes GPU-resident and
thrashing pages. Missing evidence permits no limited population. There is no
new per-block history or scan. The queue adds eight page masks (512 bytes total
on a 4 KiB-page build).

A successful limited attempt retains the block's one-shot claim. Later reports
for other pages still receive normal native servicing, but do not reopen optional
population. Eviction rearms the block. Capacity-stopped attempts retain the
existing recovery retry gate; migration-context allocation failures may retry
after releasing the reservation. This prevents a scan from repeatedly paying
reservation, mapping and completion costs for partial population of one block.
Reservations retain the conservative 2 MiB quantum; physical allocation can
also exceed the copied mask. The hardware notification itself may cover a broad
range, so this does not promise 4 KiB transfers. Byte counters measure changed
residency, not PCIe wire traffic.

Smaller copies can reduce churn but increase remote accesses, notifications or
small-transfer costs. Performance must be established on the GPU; local tests
exercise production routines with mocked hardware, not kernel loading or speed.
