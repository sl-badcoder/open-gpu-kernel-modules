# Smaller demand-paced windows and exact recovery headroom

Branch: `595.71.05/experiment/batched-lazy-eviction`.
Parent: `595.71.05/experiment/cold-only-background-eviction` (`e91238ed`).
The 8 MiB window / 32 MiB reserve policy was approved in the Lavish review.

## What changes

The parent considers a centered 64 MiB window for each new trigger and permits
concurrent speculative sessions. At the high watermark, prefetch pauses while
background eviction recovers toward the low watermark. With 90/85 watermarks
on 24 GiB VRAM, that gap is approximately 1.2 GiB.

This branch defaults to **8 MiB windows**, including the triggering 2 MiB block.
It admits **one speculative session per GPU**, held until its final tracker
wait completes. Admission returns a whole-root budget that fits below the
recovery target. A partial grant shrinks the centered window; a zero grant
does not start a session. Lazy/unpopulated and already-resident pages are still
skipped, and speculative allocation still prohibits eviction.

The new load-time parameter `uvm_perf_eager_eviction_headroom_mb=32` establishes
an occupancy target in PMA's 64 KiB page units:

```text
high   = ceil(total_pages * high_percent / 100)
low    = floor(total_pages * low_percent / 100)
target = max(low, max(high - reserve_pages, 0))
target = min(target, max(high - one_2MiB_root, 0))
```

The final clamp preserves a gap even with equal watermarks, a zero reserve,
or a very small PMA. Thus the low percentage is a floor for normal settings,
not a promise to preserve an invalid/equal-watermark configuration. Arithmetic
avoids overflowing `total_pages * percent`; free-page snapshots are clamped
to the reported total.

Prefetch may use only whole 2 MiB candidates fitting below this target.
Background eviction stops when completed plus pending root evictions would
restore it. Recovery is cleared at this **same target**, after submitted
evictions have drained. A 24 GiB PMA at the exact 90% trigger needs 16 roots
(32 MiB) to reach the default target, rather than roughly 615 roots to reach
85%. Concurrent allocations or an overshoot can require more.

Demand promotion and ordinary allocation/PMA reclaim remain outside this
speculative budget. They can consume the reserve and retain emergency reclaim.
Cold-only background selection, migration grace and the 16-root batch bound
remain. A severely overfull GPU can still need several immediate recovery
batches; this change does not impose a DtoH bandwidth limit.

Heat and migration scores are aged lazily when a root is inspected or updated,
instead of scanning every root on each aging interval. Victim selection still
scans the used list for each root; batched victim selection is not implemented.

## Retry and concurrency behavior

Budget-denied or budget-shortened attempts are counted separately and do not
set the trigger's retry bit. Space becoming available alone does not replay
the old window. A different demand trigger can start its own small window;
eviction rearms the original trigger using the existing policy.

Actual allocation-capacity stops, high-water stops during a session and
coalescing overlaps retain their existing retry behavior. These retries still
pass through the same budget before any transfer. One session per GPU makes
same-GPU window overlap uncommon; the existing bounded coalescing helpers and
lifetime cleanup remain intact.

PMA snapshots are not memory reservations: demand and other VRAM clients can
allocate concurrently. Per-block high-water checks and the no-eviction
allocation failure path remain the safety net. With eager eviction disabled,
unsupported, or valid PMA statistics unavailable, admission still serializes
sessions but grants the requested window without headroom budgeting. Invalid
statistics do not leave an old recovery state permanently latched.

## How to test

Build/install/load this branch using matching 595.71.05 userspace and GSP.
The new defaults apply automatically, but **existing explicit module options
override defaults**. In particular, an old `prefetch_chunk_mb=64` option keeps
64 MiB windows. Use these values for the approved experiment:

```text
uvm_cpu_preferred_blocks_enable=1
uvm_cpu_preferred_prefetch_chunk_mb=8
uvm_perf_eager_eviction_enable=1
uvm_perf_eager_eviction_headroom_mb=32
uvm_perf_eager_eviction_high_watermark=90
uvm_perf_eager_eviction_low_watermark=85
uvm_perf_eager_eviction_cold_threshold=0
```

The custom CPU-preferred policy remains off by default; retain `=1` for your
existing experiment. Headroom/watermarks require a module reload. The window
size remains writable at runtime. Check the running values, not just the build:

```sh
cat /sys/module/nvidia_uvm/parameters/uvm_cpu_preferred_prefetch_chunk_mb
cat /sys/module/nvidia_uvm/parameters/uvm_perf_eager_eviction_headroom_mb
cat /sys/module/nvidia_uvm/parameters/uvm_cpu_preferred_prefetch_budget_stops
cat /sys/module/nvidia_uvm/parameters/uvm_cpu_preferred_prefetched_bytes
```

`prefetch_budget_stops` counts windows denied or shortened by headroom, worker
activity, recovery or another speculative session. It is not a count of lost
demand requests. Existing capacity counters describe stops after admission;
headroom/session denials no longer inflate them.

Compare against `e91238ed`, recording loaded parameters, runtime, GPU occupancy
over time, H2D/D2H bytes and fault/notification activity. Use streaming,
sparse/uniform random, skewed hot-set, phase-changing and oversubscribed cases,
with separate cold/warm residency runs. Keep unrelated GPU work idle and
synchronize GPU completion before stopping timing. See the [shared guide](README.md).

## Validation and limits

Run all `tests/experiments/test_*.py` harnesses directly with Python. They
compile extracted production C with mocked kernel/GPU interfaces, compiler
warnings as errors, AddressSanitizer and UndefinedBehaviorSanitizer. The new
PMM harness tests exact accounting, budget admission and worker recovery;
updated coalescing tests verify partial/zero grants and no budget-stop replay.
The cold-only test continues checking victim selection, batch draining and
emergency allocation fallback.

These are not real concurrency, Linux module build/load, lockdep or PCIe
performance tests. The development host is macOS, and no application speedup
or transfer-bandwidth improvement is claimed. Eight MiB is a default, not a
hard ceiling when explicitly overridden.

This experiment bounds speculative volume and the ordinary recovery sweep;
it does not measure spare PCIe bandwidth or predict access direction. Sparse
access can still fetch unnecessary neighbors, while large sequential workloads
may lose useful lookahead. Remote counters cannot reveal all local VRAM reuse.
NVIDIA's [oversubscription study](https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/)
documents why migration policy depends on access pattern and reuse; its
[Unified Memory performance guide](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)
also explains the benefits of explicit application-directed prefetch when the
access pattern is known. A bandwidth controller or directional predictor should
be evaluated separately after measuring this branch.
