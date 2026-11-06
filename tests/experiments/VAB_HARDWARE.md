# VAB hardware experiment

This test runs on a Linux NVIDIA GPU with **this branch's modules loaded**, matching
user-space/GSP driver components, CUDA `nvcc`, and managed-memory oversubscription.
It does not change driver parameters. This is a discrete-GPU eviction experiment,
not a test of cache-hit coverage. Stop other GPU workloads so global counters can
be attributed to this process. Select one GPU with `CUDA_VISIBLE_DEVICES` and use
its matching proc file. Start with MIG disabled.

## Build and run

```sh
nvcc -O2 -std=c++14 -Xcompiler -pthread tests/experiments/vab_hardware.cu -o /tmp/vab_hardware
nvidia-smi --query-gpu=index,uuid,memory.total,memory.free --format=csv
cat /sys/module/nvidia_uvm/parameters/uvm_perf_vab_mode
cat /sys/module/nvidia_uvm/parameters/uvm_perf_vab_interval_ms
cat /sys/module/nvidia_uvm/parameters/uvm_perf_vab_eviction_linear_base
```

Choose `HOT_MIB`, `COLD_MIB`, and `PRESSURE_MIB` such that their sum exceeds available
VRAM by roughly 10–20%. Ensure sufficient available host RAM for the managed data
and OS. Make the hot allocation larger than L2 for the ordinary memory-activity
test. Large VAB granularities may require a larger hot allocation to cover several
bits. All sizes are explicit; the program never guesses a memory budget.

Example for a dedicated **24 GiB GPU**, with enough host RAM:

```sh
CUDA_VISIBLE_DEVICES=0 /tmp/vab_hardware \
  /proc/driver/nvidia-uvm/gpus/GPU-REPLACE-WITH-UUID/0/vab \
  256 4096 24000 /tmp/vab-mode3.log
```

Replace the UUID using `nvidia-smi`. The proc directory is created when UVM registers
the GPU; the test performs a managed GPU touch before opening it. The final `0`
is the non-MIG subprocessor index. A permission error is a test failure, not evidence
that hardware tracking is unsupported.

The phases are:

1. Sample idle state for at least 2 seconds and ten configured sampling intervals.
2. Fill cold memory once, then fill and repeatedly read hot memory.
3. Gradually write the pressure allocation, reading hot memory between 64 MiB writes.
4. Continue reading hot memory, then stop accesses and observe idle state.
5. Capture final counters **before any CPU data reads or cudaFree**.
6. Check every word of all three allocations on the CPU for the original pattern.

Kernel reads are volatile and contribute to a device-side result, so the compiler
cannot discard the hot scan. It is intentionally a sequence of finite kernels,
not a persistent kernel that might delay migration. Initialization and pressure
writes themselves can set VAB bits; idle and phase-labelled raw snapshots are
preserved so those effects are visible.

## What the results establish

The program reports sampling health, pressure-interval counter deltas, evidence
of victim-choice changes, completed background root frees, and full data integrity.

- `samples` increases, `enabled_bits > 0`, `sample_valid = 1`, and no new
  `dump_errors`: allocation and sampling are functioning. This alone does not prove
  that the returned bits respond correctly to the test's particular addresses.
- Compare `access_bits`, `rising_bits`, and `falling_bits` between idle and active
  phases. Repeat the run: changes provide activity evidence, but transfers and
  other accesses in the same coarse region can also set bits. A constant bitmap
  is inconclusive; it can indicate retained state, coarse coverage, or bad tracking.
- `selection_active = 1`, known eligible roots, and increased `selection_changes`:
  VAB caused the production picker to choose a different root from its baseline.
- Increased `background_roots_freed`: background eviction released physical GPU
  root allocations. This counter increments **after** `free_root_chunk`, whose
  path waits for pending migration. It is stronger evidence than a victim-selection
  count, which increments before eviction succeeds. Roots are 2 MiB here.
- Integrity passes: evicted contents were preserved. Eviction removes VRAM
  residency; it must not erase the logical data. CPU validation can itself migrate
  data, which is why it occurs after final measurement.

Exit status: `0` = healthy sampling, changed selection, completed background frees,
and intact data observed; `2` = integrity and sampling passed but VAB-driven eviction
evidence was incomplete; `1` = CUDA, I/O, sampling-health, or integrity failure.
Neither status 0 nor timing alone proves that a **specific cold allocation** was
evicted instead of a hot one: the current counters contain no address attribution.
An inactive selection result is expected if the physical mapping is unconfigured.

The linear-base parameter is read-only at runtime and defaults to an unset value.
Do not set it to zero merely to make this test pass. Establish the hardware mapping
first; a guessed base can produce plausible but incorrect victim choices.

## Control run

Use mode 1 (observe, baseline victim policy) versus mode 3 (VAB recency), with the
same workload sizes, sample interval, mapping configuration, and watermark settings.
Use fresh workload processes and repeated runs in alternating order.

On the target host, save the original mode before changing it:

```sh
cat /sys/module/nvidia_uvm/parameters/uvm_perf_vab_mode
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_vab_mode
# Run the workload above, writing /tmp/vab-mode1.log; exit status 2 is expected.
echo 3 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_vab_mode
# Run it again, writing /tmp/vab-mode3.log.
# Restore the original mode when finished.
```

Mode 1 can still perform emergency/watermark eviction. Consequently, frees in mode
1 are not evidence of VAB-guided eviction. Mode 3 can also fall back to the baseline;
inspect `selection_fallbacks`, `selected_unknown`, and sample freshness.
If pressure never causes frees or eligible roots, increase pressure, check the
loaded driver/configuration, and inspect the log; do not interpret it as a pass.

For **allocation-specific causal proof**, add or use tracing that records the
selected physical root, its mapped virtual range/owner, sample identifier, VAB
classification/recency, baseline candidate, and completed eviction/free. Correlate
that with hot/cold allocation ranges and compare modes 1 and 3. The current proc
interface cannot provide that stronger assertion; this experiment does not invent
per-allocation residency from `cudaMemRangeAttributeLastPrefetchLocation` or latency.

## Sampling and resetting

`uvm_perf_vab_interval_ms` changes how frequently the worker requests CURRENT dumps
(5–1000 ms, default 100 ms). It is **not a verified hardware-reset control**. Scheduling
and dump duration also make the actual cadence approximate.

The public controls expose CURRENT/DIFF/AGGREGATE and enable/disable logging, but no
explicit general-purpose clear/rearm operation. CURRENT's comment promises current
state and clearing prior aggregation, not a documented fresh hardware epoch.
`DISABLE_MODE_CLEAR` concerns protected-mode behavior; it is not a periodic reset API.
Clearing `currentBits` or the cached sample in software does not clear hardware state.

Before implementing windowed recency, establish a supported dump-and-clear/rearm
contract for the exact GPU, including completion ordering and effects on translation
and caches. A controlled no-pressure test should record: idle baseline, one known
region touch, repeated idle dumps, and a second touch of that same resident region.
Use a verified address-to-bit mapping and isolate unrelated traffic. If bits fall
without new touches and set again on retouch, that supports rearming, but does not
by itself prove exact epoch boundaries or cache-hit coverage. Do not create windows
by repeatedly allocating/freeing the RM resource without a supported contract.

With **verified independent windows**, the intended recency algorithm is:

```text
at completed window k:
    for every enabled bit set in window[k]:
        last_seen[bit] = window[k].timestamp
    leave other last_seen timestamps unchanged
    rank a root by the newest last_seen of the bits it covers
```

Consecutive windows `1, 1, 1` then refresh three times. The current implementation
instead timestamps newly observed set bits (`bits & ~(previous & common)`). It
would miss refreshes for a continuously active region even if CURRENT already
produces fresh windows. Change that only after verifying the hardware contract;
timestamping every set bit in a sticky bitmap would falsely make old data fresh.
Keep coverage, sample freshness, ownership-generation, and residency-readiness
checks. Resolution is one sampling window at best, and multiple allocations can
share a VAB region, so this remains approximate recency rather than exact LRU.

## Validation limits

The source is intended for `nvcc` on the target Linux/CUDA machine. Host-only mocked
tests in this directory exercise software policy but cannot validate hardware bits,
cache coverage, reset semantics, or physical eviction. The development Mac has no
CUDA compiler/GPU; the hardware experiment has not been executed there.
