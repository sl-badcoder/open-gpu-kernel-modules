# Staged vLLM sensitivity tests

Use `run_vllm_sensitivity.py` from the vLLM fork directory. It reloads the existing
`nvidia-uvm.ko` before every independent run, then calls the monitored benchmark.
No new kernel changes or additional module are required. Build the bounded-refill
branch first. Stop other CUDA clients before starting; unload failure stops the
suite rather than forcing removal. Keep your external VRAM reservation and host
workload constant across comparisons. A persistent reserver must permit UVM
unload; the runner does not kill it. Do not run suites concurrently. Prefer
`--gpu GPU-<UUID>` if CUDA ordinals and physical indices may differ.

The runner fixes model Qwen2.5-3B-Instruct, managed KV12GiB, 32 prompts,
10240 input/256 output tokens, hardware threshold1024, base admission2,
population cap32MiB, eager interval25ms, high/low99/90, VAB3 and linear base0.
Use the same verified VAB mapping as previous runs. Refault, sparse and the old
backoff/recovery controllers are off. The runner uses only registered module
parameters: older loader names `uvm_perf_prefetch_min_signal_density`,
`uvm_cpu_preferred_region_migration_enable` and
`uvm_perf_migration_aware_eviction_enable` are absent from this branch and omitted.
Selected `--gpu` also sets CUDA_VISIBLE_DEVICES for the benchmark, while the
monitor uses the same physical index or UUID.

Start with a plan only (no sudo, reload or benchmark execution):

```bash
python3 ../../open-gpu-kernel-modules/tests/experiments/run_vllm_sensitivity.py \
  --phase admission --batches 32 --rounds 3
```

Add `--execute --module ../../open-gpu-kernel-modules/kernel-open/nvidia-uvm.ko`
to execute. The path may differ on your machine. The runner uses sudo only for
rmmod/insmod; launch it as your normal user. It verifies the loaded settings and
records the module file SHA256. It leaves the last configuration loaded.

## Recommended order

| Phase | Configurations | Purpose |
|---|---|---|
| `admission` | Legacy; bounded extra0; bounded extra2 | Isolate refill improvement and whether extra probation still helps |
| `refill` | Batch4, 8, 16: refill8, 16, 32MiB | Trade refill frequency against burst size |
| `reserve` | Reserve32, 64, 128MiB | Check whether withheld headroom helps native demand or limits useful residency |
| `pressure` | Admission80, 85, 90% | Test when clear-separated evidence should be required |
| `validation` | One chosen setting, batches8/24/32 | Check that gains survive lower memory pressure |

Start with admission on batch32. Three configurations × three independent runs
means **nine runs, each containing one cold and two warm passes**. At the recent
batch32 timings this is roughly an hour, potentially longer. `--rounds 1` is a
three-run screening test; it cannot establish repeatability. Order is shuffled
within each independent round using a recorded deterministic seed.

Use the winner's extra setting in the following phases. For example, if extra0
wins, run:

```bash
python3 ../../open-gpu-kernel-modules/tests/experiments/run_vllm_sensitivity.py \
  --phase refill --batches 32 --rounds 3 --extra 0 \
  --execute --module ../../open-gpu-kernel-modules/kernel-open/nvidia-uvm.ko
```

If batch8 wins that phase, carry it into the reserve comparison:

```bash
python3 ../../open-gpu-kernel-modules/tests/experiments/run_vllm_sensitivity.py \
  --phase reserve --batches 32 --rounds 3 --extra 0 --batch-count 8 \
  --execute --module ../../open-gpu-kernel-modules/kernel-open/nvidia-uvm.ko
```

Then test pressure with the chosen `--reserve-mb`, `--batch-count` and `--extra`.
Finally use `--phase validation --batches 8 24 32 --rounds 3`, passing all chosen
settings explicitly. Do not change threshold or KV capacity during these stages.
This is a staged search, not a full factorial study; parameter interactions may
require a final matched confirmation against the original reference at all three
batch sizes. Stop after the useful stages rather than sweeping every combination.

## Outputs and interpretation

Every suite and run has a unique folder. `plan.json` records jobs and module hash;
each run has `sensitivity.json` with requested/actual parameters, plus the normal
before/after snapshots, samples, histogram, log and benchmark. `--interval 2` is
the default. Offline dmon is disabled unless `--pcie` is passed, reducing monitor
overhead; hold this choice fixed across comparisons.

`runs.csv` has one row per independent run. `aggregate.csv` groups matched
settings and reports median/min/max across independent runs. The warm throughput
is total warm output tokens divided by total warm time, not an unweighted average
of rates. Cold throughput remains separate. Failed/incomplete runs are labelled
and excluded from aggregates; a benchmark failure stops further execution.
Reload failures are also recorded and stop execution. `paired.csv` compares
configurations within the same batch and independent repeat, reducing drift:
references are bounded-extra0 for admission, batch16 for refill, reserve64 for
reserve, and pressure85 for pressure. Ratios above1 mean higher candidate
throughput or higher candidate movement, respectively. Validation has no paired
reference. Inspect these paired ratios as well as independent medians.

Compare:

- Warm tokens/s median and min/max: require a repeatable gain, not just a peak.
- Population MiB per output token and releases per1000 output tokens: avoid
  mistaking shorter runtime alone for less movement.
- Capacity stops and replacement requests: denied attempts can rise while
  throughput improves; these are not uniquely avoided migrations or work kicks.
- Return-distance byte fractions: show which cyclic returns survive admission.

Population/release totals include model initialization, the cold pass, warm
passes and teardown. Their per-token normalization is a matched full-run metric,
not warm-only traffic. Population bytes are residency accounting, not PCIe wire
bytes. Root releases are allocator events. Counters are driver-wide; keep other
GPU/host activity stable. Two warm passes in one process are not two independent
trials. A one-factor sweep can identify a useful setting without proving a global
optimum.

To rebuild CSVs later without reloading anything:

```bash
python3 ../../open-gpu-kernel-modules/tests/experiments/run_vllm_sensitivity.py \
  --summarize results/uvm-sensitivity/<suite-folder>
```

Send the complete suite folder, including failures, rather than selecting the
fastest benchmark file. Raw data remain available for per-window plotting.
