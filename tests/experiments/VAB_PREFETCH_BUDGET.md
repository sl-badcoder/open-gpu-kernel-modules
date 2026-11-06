# VAB prefetch budget experiment

Branch: `615.71.09/experiment/vab-prefetch-budget`, based on
`615.71.09/experiment/vab-lru-no-decay`.

This experiment reduces speculative migration pressure. It does not reset VAB,
change the hardware address mapping, or claim to measure cache hits or exact reuse.

## Policy

1. Every access-counter-guided chunk prefetch reserves bytes before submission.
   Reservations are shared by concurrent service threads on the GPU and remain
   charged until the chunk's final GPU tracker wait, including error paths.
   Both the fixed ceiling and adaptive window limit the **aggregate** reservations.
   The effective chunk can be smaller than `uvm_cpu_preferred_prefetch_chunk_mb`.
2. Admission also requires current PMA free capacity minus existing reservations
   and a demand reserve. Already-pending frees are not credited. The free-capacity
   check is conservative: PMA can already exclude allocations represented by a
   reservation. Demand can race and consume capacity; this is an admission check,
   not a permanently dedicated physical pool. Speculative allocations continue to
   use `retry.allow_eviction = false`.
3. Background eviction of used roots requires a fresh, mapped, ownership-ready
   VAB-COLD classification and the residency grace period. Neither HOT nor UNKNOWN
   used roots are proactively evicted, even above the high watermark. Free,
   unused, and discarded chunks retain their existing priority; they do not
   require preserving live contents through a new copy. Mandatory allocation and
   PMA reclaim retain their fallback and can reclaim hot or unknown memory.
4. If a newly copied speculative **neighbor block** is evicted within the feedback
   window and receives a new demand/access-counter signal within that window
   after eviction, halve the adaptive window. The timestamp belongs to the
   virtual block/GPU state, so physical-root reuse cannot transfer this history
   to a different allocation. Consume each return marker once. Ordinary demand
   blocks and neighbor prefetch submissions are not counted as return signals.
5. Actual capacity failures, insufficient admission headroom, and watermark
   recovery also reduce the window. Merely colliding with an active background
   worker or another prefetch's byte reservation does not. Reductions are limited
   to one per feedback interval and stop at 4 MiB (or the fixed cap if smaller).
6. After a quiet interval, a successful nonempty attempt can increase the window
   by 2 MiB, at most once per interval, up to the fixed cap. This is a bounded
   probationary probe, **not proof of useful reuse**. Empty or failed attempts do
   not recover. The 4 MiB floor leaves one neighbor slot even if demand promotion
   made the trigger a no-op. An explicitly configured 2 MiB hard cap intentionally
   permits only the trigger block.

Reservations cover full 2 MiB VA blocks even when only some pages migrate; sparse
ranges may therefore be throttled earlier than their copied payload requires.
The trigger block is serviced before neighbors and is conservatively charged to
this callback's reservation too. Direct demand service runs before this callback
and is not subject to the speculative budget. Explicit application CUDA prefetch
and other migration mechanisms are outside this experiment's admission point.

The byte limit bounds outstanding speculative payload, not sustained bytes/sec.
It also does not bound allocator batching, page-table allocations, or compulsory
copies. Rapid eviction/return is a churn heuristic: it can identify an expensive
replacement cycle, but cannot prove that the evicted data was never used. Pressure
feedback can shrink speculation even when no return is observed.

## Parameters

These new parameters are set when loading `nvidia_uvm`, using your normal module
installation/reload procedure (loading an already loaded module will not change
them):

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `uvm_perf_speculative_prefetch_limit_mb` | 32 | Hard per-GPU outstanding reservation ceiling in MiB; rounded down to 2 MiB. Values 0 or 1 disable this speculative path without retry churn. |
| `uvm_perf_speculative_prefetch_reserve_mb` | 64 | PMA free capacity withheld from speculative admission, in MiB. |
| `uvm_perf_speculative_prefetch_quiet_ms` | 1000 | Prefetch-to-eviction and eviction-to-return windows; minimum spacing between backoffs and recovery probes. Effective minimum 1 ms. |

Keep `uvm_cpu_preferred_blocks_enable=1` and a nonzero
`uvm_cpu_preferred_prefetch_chunk_mb` if testing the access-counter-guided chunk
path. Existing block-promotion and other policy settings should match the control
run. For example, load-time experiment arguments can include:

```text
uvm_cpu_preferred_blocks_enable=1
uvm_perf_speculative_prefetch_limit_mb=32
uvm_perf_speculative_prefetch_reserve_mb=64
uvm_perf_speculative_prefetch_quiet_ms=1000
```

VAB background selection still requires a **verified** physical range mapping
through `uvm_perf_vab_eviction_linear_base`. Do not guess a base to enable selection.
With unknown mapping, used-root background eviction stops; demand/PMA reclaim can
still progress and capacity feedback still throttles speculative admission. VAB
window/reset semantics remain unchanged. Suspect VAB coverage must not be treated
as reliable evidence of coldness.

## Measurements

Read the selected GPU's `/proc/driver/nvidia-uvm/gpus/GPU-UUID/0/vab` while running
the same oversubscribed workload on the control and experiment branches. Use
[VAB_HARDWARE.md](VAB_HARDWARE.md) for the general pressure/integrity test; for this
policy specifically, also use a workload which generates access-counter-triggered
neighbor prefetches. Confirm `prefetch_admissions` increases, otherwise that run
did not exercise the new budget. Read these new fields:

- `background_policy VAB_cold_only`
- `background_hot_skips` / `background_unknown_skips`: visits skipped by the
  proactive used-root picker, not distinct-root counts.
- `prefetch_limit_bytes`, `prefetch_demand_reserve_bytes`: configured settings.
- `prefetch_reserved_bytes`, `prefetch_peak_bytes`: live/peak reservations.
- `prefetch_window_bytes`: adaptive aggregate ceiling; zero before first use.
- `prefetch_admissions`, `prefetch_denials`: admission outcomes.
- `prefetch_backoffs`, `prefetch_quick_returns`, `prefetch_capacity_stops`,
  `prefetch_recoveries`: feedback and adjustment counters.

Check that peak reservations never exceed the fixed cap and that reservations
return to zero after work drains. When backoff lowers the window below already
reserved bytes, existing transfers finish normally and new admissions wait for
the count to drain. Immediate cancellation is neither required nor attempted.

Induce a quick eviction-and-return cycle and check that `prefetch_quick_returns`
and `prefetch_backoffs` rise and the window falls toward 4 MiB. A cold streaming
workload may instead back off via `prefetch_capacity_stops`. The existing
`selected_hot` counter also includes mandatory demand selections; it need not be
zero. Use the production cold-only picker test to validate that background
selection cannot select a known hot used root.

Record total inbound/outbound migration bytes, quick-return rate, demand stalls,
and application runtime with your profiler. Compare repeated runs against the
parent branch with identical workload, sample interval, mapping, and watermarks.
Also compare caps such as 8, 16, and 32 MiB. Lower traffic with worse application
runtime is not automatically an improvement. This policy does not promise to
remove compulsory traffic when the useful working set exceeds VRAM.

## Software validation

```sh
python3 tests/experiments/test_prefetch_budget.py
python3 tests/experiments/test_speculative_prefetch.py
python3 tests/experiments/test_demand_first_eviction.py
python3 tests/experiments/test_demand_first_notifications.py
python3 tests/experiments/test_vab_classifier.py
python3 tests/experiments/test_vab_observation.py
python3 tests/experiments/test_vab_readiness.py
python3 tests/experiments/test_vab_recency.py
python3 tests/experiments/test_legacy_policy_port.py
```

These execute production policy code with mocked kernel/RM dependencies and
address/undefined-behavior sanitizers. They cannot validate hardware VAB coverage,
reset semantics, real interconnect traffic, or a complete Linux module build.
Build the modules against the target Linux kernel and use matching 615.71.09
user-space/GSP components before hardware evaluation.
