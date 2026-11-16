# Monitored vLLM experiments

Run the wrapper from your vLLM fork's working directory (the directory containing
`.venv` and `benchmarks/benchmark_uvm.py`):

```bash
bash ../open-gpu-kernel-modules/tests/experiments/run_vllm_monitored.sh
```

The wrapper uses your exact benchmark arguments, including three repeats. It
creates a unique directory under `results/ember-adaptive/` for each batch size.
Existing experiment directories are never overwritten. No kernel parameters are
changed and the module is not reloaded. Python 3's standard library is sufficient.

Optional overrides:

```bash
GPU_ID=GPU-8289d291-5c6b-39a8-3b15-1854938bbc3e \
BATCH_SIZES="8 16 24" SAMPLE_INTERVAL=2 RESULT_ROOT=results/threshold-128 \
bash ../open-gpu-kernel-modules/tests/experiments/run_vllm_monitored.sh
```

`GPU_ID` selects the physical GPU for NVIDIA telemetry; it does not change the
benchmark's CUDA device selection. Set `CUDA_VISIBLE_DEVICES` consistently if
needed. VAB snapshots collect all readable GPUs and sysfs counters are global.
Avoid concurrent GPU workloads when comparing configurations.

## Saved files

- `benchmark.json`: the benchmark's normal performance results.
- `benchmark.log`: combined benchmark stdout/stderr, also displayed live.
- `metadata.json`: exact command, working directory, host/kernel version, selected
  GPU, CUDA visibility, driver version, NVIDIA inventory and source Git commits
  when available. Source commits do not certify which kernel build was loaded.
- `before.json`, `after.json`: all readable UVM module parameters; all VAB
  counters, settings and raw bitmaps; host memory/swap/VM statistics and cumulative
  CPU time ticks; GPU memory,
  utilization, clocks, temperature, power and PCIe link configuration.
- `run_status.json`: starting, running, finalizing or finished, with the monitor
  PID and exit code when available. A stale state after an abrupt kill is possible.
- `last_sample.json`: atomically replaced checkpoint of the latest completed
  sample. It is useful after a killed run, but is not an after-run measurement.
- `samples.csv`: timestamped long-form metrics every two seconds by default,
  with interval changes/rates for known cumulative UVM counters.
- `samples.jsonl`: full snapshots including raw text and collection errors.
- `changes.csv`: before/after numeric changes. Only known cumulative counters
  receive rates; gauges/settings retain signed changes. A decreasing counter is
  marked as a reset and its change/rate is omitted.
- `observed_changes.csv`: first/last available numeric readings and their coverage
  times, including GPU counters that appear after CUDA initialization or disappear
  at teardown. These intervals may cover only part of inference; unobserved events
  are not reconstructed. A single reading has no meaningful rate.
- `summary.json`: exit status, elapsed snapshot interval, changes and derived
  short-return fraction, paused-plan fraction, hot-skip visits per freed root.
  Derived ratios use the observed intervals; `vab_seen` says whether any VAB
  readings were captured at all.
- `pcie_dmon.log`: continuous NVIDIA PCIe RX/TX telemetry with dates/times and
  tool-provided units. Unsupported GPUs/drivers leave their error here. This is
  total GPU PCIe traffic, not exclusively KV migration and not NVLink traffic.

Sampling is read-only, but proc VAB reads scan PMM roots and NVIDIA queries have
some overhead. Start with two-second sampling; increase to five seconds if the
monitor changes throughput materially. `collection_seconds` records collection
cost. Samples are sequential rather than atomic; actual spacing can exceed the
requested interval. This does not change the kernel's VAB sampling interval.
Optional unsupported NVIDIA fields are omitted; missing sources are reported.
An initial missing-VAB warning can be normal before a CUDA/UVM GPU registration.
The collector searches again on every sample. If `summary.json` reports
`vab_seen: false`, check the proc tree during the workload:

```bash
ls -la /proc/driver/nvidia-uvm/gpus/
cat /proc/driver/nvidia-uvm/gpus/GPU-YOUR-UUID/0/vab
```

Check saved discovery/read errors for permissions or a missing proc tree. A missing
`vab` file while the GPU is registered can also indicate that the loaded module
does not contain this experiment. The monitor cannot provide VAB statistics that
the loaded driver does not expose. Running the entire benchmark as root is not
required merely because the pre-context snapshot has no GPU entries.
Use `monitor_uvm.py --no-pcie` to disable the separate one-second PCIe sampler.

The complete interval includes model loading, warmup, all three repeats and
teardown. It cannot identify individual inference passes without timestamps from
the benchmark. Do not divide these whole-run counters by inference-only time and
call that inference bandwidth. Driver byte counters are bookkeeping, not exact
PCIe wire bytes. Short-return counts are events, not unique pages. A counter reset
that surpasses the previous value before the next sample cannot be detected.

`after.json` is created when the benchmark command exits, including a nonzero
exit or a handled Ctrl-C/SIGTERM. It will not exist while the benchmark is still
running. SIGKILL, killing the monitor via the OOM killer, or a host failure can
prevent finalization; inspect `last_sample.json` and `samples.jsonl` in that case.
The final snapshot is saved before CSV/summary processing. Disk errors and other
unexpected finalization failures can still prevent output; retain the terminal
traceback when reporting missing files.

For clean per-repeat comparisons, use `--repeats 1` and launch separate monitored
runs. To distinguish loading/warmup from inference, retain benchmark timestamps or
add phase markers in the vLLM benchmark; that source is outside this repository.

## Comparing runs and plotting

Compare completed tokens per second alongside demand bytes/token, speculative
bytes/token, short-return fraction, eviction counts, and PCIe RX/TX. Use actual
completed token counts from `benchmark.json`, rather than requested tokens. Keep
the workload and eviction settings identical when sweeping access thresholds.
The current kernel does not expose every hardware counter crossing or promotion
latency; collecting existing statistics cannot recover those measurements.

Example plot, using optional pandas and matplotlib in your analysis environment:

```python
import pandas as pd
import matplotlib.pyplot as plt

run = "results/ember-adaptive/YOUR-RUN-DIRECTORY"
df = pd.read_csv(f"{run}/samples.csv")
gpu = "GPU-8289d291-5c6b-39a8-3b15-1854938bbc3e"
metrics = {
    f"vab/{gpu}/0/prefetch_refault_demand_bytes": "Demand GiB/s",
    "module/uvm_cpu_preferred_prefetched_bytes": "CPU-policy prefetch GiB/s",
}
for metric, label in metrics.items():
    rows = df[df.metric == metric]
    rate = pd.to_numeric(rows.counter_per_second, errors="coerce") / 2**30
    plt.plot(rows.elapsed_s, rate, label=label)
plt.xlabel("Seconds since before snapshot (includes initialization)")
plt.ylabel("Driver-accounted GiB/s")
plt.legend()
plt.tight_layout()
plt.savefig(f"{run}/movement.png", dpi=160)
```

Generic usage for another command:

```bash
python3 ../open-gpu-kernel-modules/tests/experiments/monitor_uvm.py \
  --output-dir results/my-run --gpu 0 --interval 2 -- \
  .venv/bin/python benchmarks/benchmark_uvm.py ...
```
