# Rapid-return ratio prefetch experiment

Branch: `615.71.09/experiment/rapid-return-ratio`, based on
`615.71.09/experiment/vab-prefetch-budget` at `6e0e809f`.

This experiment uses **one backoff metric**: the fraction of speculative neighbor
block population episodes followed by rapid eviction and rapid demand return.
It removes the previous allocation-failure confirmation filter, copy-completion
recovery credits, and sequential-growth detector. Recovery now uses the same
mature rapid-return ratio as backoff, with bounded probes when OFF.
It does not change VAB sampling, address mapping, or reset behavior.

## Metric and tracking

For each speculative neighbor VA block where `make_resident` succeeds and reports
nonzero changed-residency bytes, count one population episode and attach a cohort
token to that block's GPU state. The triggering block, already resident neighbors,
missing VA blocks, and allocations that fail without moving data do not count.
Sparse and full blocks each count as one; this is not a byte-weighted ratio.

A numerator event requires both:

1. An eviction to CPU within `T` of the speculative population timestamp.
2. A demand access-counter callback for that block within `T` of that eviction.

`T` defaults to **20 ms**. It is an event age limit, **not a sampling or polling
period**. Timestamps are taken at submission/residency bookkeeping, not GPU copy
completion, so queued transfers consume some of this opportunity. Population
records remain provisional until every contributing prefetch attempt finishes
successfully, including GPU mapping, dependency collection, and final tracker
waiting. A failed attempt discards each cohort it contributed to; other successful
records in that same cohort are conservatively discarded too. Fixed references
in each reservation account for outstanding records without allocating memory.
A cohort cannot mature while any contributing attempt is still in flight. Long
attempts can therefore delay evaluation and occupy counter slots; slot collisions
are counted as dropped records rather than overwriting unfinished evidence.
The demand callback runs after demand service and before the one-shot prefetch
gate, including when admission will be denied. The per-block token is consumed
once. Additional partial evictions preserve the first eviction timestamp rather
than extending the return deadline. Another speculative population replaces the
block's previous tracking episode; superseded episodes retain their denominator
entry but cannot later contribute a return. Tracking is deliberately coarse at
VA-block/GPU granularity, not individual migrated pages.

A rapid return indicates an expensive replacement cycle. It does **not** prove
that a block was unused, or that prefetch caused its eviction. An unused block
that never returns contributes no numerator event. Mandatory demand movement and
explicit application CUDA prefetch are outside this population denominator.

## Cohorts and decisions

A GPU keeps four fixed counter slots for `T`-wide population cohorts. A cohort
collects populations for one interval and then waits two complete intervals.
This gives even its youngest member the full two-event observation opportunity.
With `T=20 ms`, an episode enters the mature denominator after **40–60 ms**,
plus any delay until contributing attempts complete and the next accounting/
admission event. Idle GPUs run no timer.
Newly populated cohorts never dilute mature results.

Mature cohorts accumulate into a decision batch until there are at least
**32 blocks**. Then:

```text
ratio = mature rapid-return episodes / mature population episodes

ratio >= 50%: reduce the adaptive byte window by approximately 25%
ratio <  50%: increase an active window by 2 MiB, up to the configured cap
              (OFF reopens at 4 MiB after sufficient mature probe evidence)
```

Every decision consumes its entire batch, even below threshold. On any size change,
all still-immature cohorts are discarded: another change requires populations
recorded under the new size. Unique cohort IDs reject late old-policy returns,
even if new populations arrive in the same time interval. Reservations also carry
a generation: remaining populations submitted by an older in-flight attempt are
discarded from measurement after a size change, while its transfers drain normally.
A decision batch cannot cause another change. A large batch causes at most one
step, not one step per group of 32. Sub-minimum batches remain pending across
idle periods; they have no additional expiration timeout. Traffic from several
allocations on the same GPU shares the ratio and budget.

At the default 32 MiB cap, successive reductions are:

```text
32 -> 24 -> 18 -> 14 -> 12 -> 10 -> 8 -> 6 -> 4 -> OFF   MiB
```

Recovery is enabled by default. An active window grows **2 MiB per mature batch
strictly below the configured threshold**. Equality still backs off. There is no
separate sequential-growth path, elapsed-time growth credit, or reward for copy
completion. At the cap, a good batch changes neither size nor generation. Because
there is no hysteresis band, ratios fluctuating around the threshold can alternate
small increases and reductions; the minimum batch size controls evidence volume.

OFF has no normal speculative populations to observe. After older reservations
drain, admit at most one **4 MiB probe** (trigger plus neighbor) at a time, no sooner
than **3T after entering OFF**, then **3T after each probe completes**. With T=20 ms,
this cooldown is 60 ms; it derives from the full cohort observation horizon rather
than a separate timing parameter. Probes obey the same cap and headroom checks;
a partial grant smaller than 4 MiB is refused. A 2 MiB cap or a one-block request
cannot probe and remains OFF. Failed attempts have their provisional cohorts discarded. Empty and trigger-only
probes provide no neighbor evidence. None can cause recovery merely by completing;
only completed successful neighbor populations enter the mature denominator.

Each actual neighbor populated by a probe contributes to the usual mature batch.
Only a batch of at least `min_blocks` with a ratio below the threshold reopens the
window at 4 MiB. A bad probe batch leaves it OFF and bounded probing continues.
With min_blocks=32 and one neighbor per probe, approximately 32 probes are needed:
about two seconds plus GPU work and demand-notification delays at default T.
No notifications, no headroom, or too few observed neighbors means no recovery.
This path can still move unused data: low ping-pong is not proof of useful reuse.

Set `uvm_perf_speculative_prefetch_recovery_enable=0` to retain the preceding
experiment's one-way backoff and permanent OFF. Set
`uvm_perf_speculative_prefetch_backoff_enable=0` to disable both reductions and
recovery/probes while preserving fixed-cap prefetch and ratio diagnostics. These
are module-load controls, not runtime reset switches. Reload between independent
runs: budget and evidence are per GPU, not per process.

## Admission and VAB eviction

The adaptive window and fixed cap bound aggregate reserved speculative bytes per
GPU, including concurrent service threads. Reservations use full 2 MiB units and
remain live through the final GPU tracker wait, including error paths. Already
reserved work may finish after a reduction; new admission waits until it fits.
The triggering block is visited first and conservatively charged to the
reservation. Demand service happens before this callback and is not disabled by
OFF. The actual chunk is also bounded by `uvm_cpu_preferred_prefetch_chunk_mb`.

Admission preserves 64 MiB of free PMA capacity by default and subtracts existing
reservations conservatively. It is a headroom check, not an exclusive physical
pool; demand can race with it. Allocation failures, watermark recovery, and worker
activity can stop or defer speculation, but **cannot reduce the adaptive window**.
Allocation failures remain diagnostic counters. Speculative allocations still
use `retry.allow_eviction = false`.

Background eviction still protects HOT and UNKNOWN used roots and requires fresh,
mapped, ownership-ready VAB-COLD evidence. Mandatory demand/PMA reclaim keeps its
fallback to make progress. The default background residency grace remains
**100 ms**, which exceeds the 20 ms rapid-eviction cutoff: ordinary background
selection generally cannot produce this metric's numerator. Mandatory reclaim
can. Keep this distinction in mind when interpreting a zero ratio. Changing the
grace is a separate experimental variable, not an implicit part of this branch.

The cap bounds in-flight payload, not sustained interconnect bytes per second.
It does not bound compulsory traffic, page-table allocations, or allocator batching.

## Controls

All these parameters are set at module load and read-only afterward.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `uvm_perf_speculative_prefetch_backoff_enable` | 1 | Set 0 to disable all adaptive changes and probes; retain fixed-cap prefetch, headroom checks, and ratio diagnostics. |
| `uvm_perf_speculative_prefetch_recovery_enable` | 1 | Grow on mature batches below threshold and probe from OFF. Set 0 for one-way backoff. Requires backoff_enable=1. |
| `uvm_perf_speculative_prefetch_limit_mb` | 32 | Aggregate speculative cap, rounded down to 2 MiB. Below 2 disables speculation. |
| `uvm_perf_speculative_prefetch_reserve_mb` | 64 | Free PMA capacity excluded from admission. |
| `uvm_perf_speculative_prefetch_churn_ms` | 20 | Each observation interval T; minimum 1 ms. |
| `uvm_perf_speculative_prefetch_return_percent` | 50 | Backoff percentage, clamped to 1–100. Equality triggers. |
| `uvm_perf_speculative_prefetch_min_blocks` | 32 | Minimum mature population episodes per decision; minimum 1. |

The previous `uvm_perf_speculative_prefetch_quiet_ms`,
`uvm_perf_speculative_prefetch_backoff_ms`, and
`uvm_perf_speculative_prefetch_recovery_ms` parameters have been removed. Remove
them from old module-load commands. Existing VAB and watermark controls remain.

To compare adaptive backoff against fixed-cap prefetch, add
`uvm_perf_speculative_prefetch_backoff_enable=0` to the module-load command.
Use `=1` (the default) for adaptive backoff/recovery. This switch is read-only after loading;
unload/reload between runs. Both modes retain ratio bookkeeping, the same fixed
byte ceiling, and demand-headroom checks. To disable speculative prefetch entirely,
use `uvm_perf_speculative_prefetch_limit_mb=0` instead.

Example additions to the usual matching-module load command:

```sh
uvm_perf_speculative_prefetch_limit_mb=32 \
uvm_perf_speculative_prefetch_reserve_mb=64 \
uvm_perf_speculative_prefetch_churn_ms=20 \
uvm_perf_speculative_prefetch_return_percent=50 \
uvm_perf_speculative_prefetch_min_blocks=32
```

## Diagnostics and evaluation

Read the existing per-GPU VAB proc file, for example:

```sh
cat /proc/driver/nvidia-uvm/gpus/GPU-REPLACE-WITH-UUID/0/vab
```

Relevant fields:

- `prefetch_policy`: `rapid_return_ratio`.
- `prefetch_backoff_enabled`: 1 for adaptive operation, 0 for fixed-cap operation.
- `prefetch_recovery_enabled`: effective recovery setting (requires backoff enabled).
- `prefetch_recoveries`, `prefetch_probes`, `prefetch_probe_inflight`: window
  increases and OFF probe activity. `prefetch_next_probe_ns` is the earliest
  next-probe timestamp, not an automatic wakeup; it is relevant while OFF only.
- `prefetch_observation_ns`, `prefetch_return_percent`, `prefetch_min_blocks`:
  effective configuration.
- `prefetch_prefetched_blocks`, `prefetch_quick_returns`: lifetime populations and
  accepted returns, including immature cohorts. Do not divide these to reproduce
  the controller's last decision.
- `prefetch_mature_blocks`, `prefetch_mature_returns`: lifetime mature totals.
- `prefetch_unretired_blocks`, `prefetch_unretired_returns`: totals still in
  cohort slots. `prefetch_ready_blocks` and `prefetch_ready_returns` are the
  subset old enough and fully completed to retire at snapshot time, awaiting the
  next event. These
  expose the final cohort after a short workload without changing the policy.
- `prefetch_pending_blocks`, `prefetch_pending_returns`: mature evidence below
  the minimum batch size.
- `prefetch_last_blocks`, `prefetch_last_returns`: numerator/denominator of the
  last evaluated batch. Divide last_returns by last_blocks when nonzero.
- `prefetch_discarded_blocks`, `prefetch_discarded_returns`: immature evidence
  discarded on a size change or attempt failure, plus old-generation populations
  still in flight.
  These populations remain in the lifetime population
  total but never enter the mature denominator.
- `prefetch_dropped_records`, `prefetch_dropped_returns`: delayed slot collisions
  and unmatched/expired tokens, including old-policy tokens discarded at backoff.
- `prefetch_ratio_evaluations`, `prefetch_backoffs`, `prefetch_window_bytes`,
  `prefetch_off`: decisions and resulting state.
- `prefetch_reserved_bytes`, `prefetch_peak_bytes`, `prefetch_admissions`,
  `prefetch_denials`, `prefetch_capacity_stops`, `prefetch_pressure_pauses`,
  `prefetch_allocation_failures`: admission diagnostics.

Proc reads are passive: cohorts retire on accounting/admission/completion events, not when
statistics are read. During idle time a snapshot can therefore show unretired
cohorts. Sequential-growth telemetry remains removed.

Ratios normalize the number of moved blocks, but do not remove all tuning. T
still depends on migration latency, work rate, hardware, and eviction policy;
32 blocks controls statistical stability and response delay; the 50% threshold
expresses a policy choice. A block percentage also hides differences in payload
size. Sweep T (for example 10, 20, and 50 ms) with fixed threshold/minimum and
identical eviction settings, reloading between runs. Record runtime, both
migration directions, demand stalls, cohort counts, and the window trajectory.
This environment cannot measure GPU or interconnect performance.

A future progress-based experiment could define the return horizon as a fraction
of VRAM worth of intervening migration bytes instead of elapsed milliseconds.
That removes a wall-clock cutoff but still chooses a distance threshold and must
mature cohorts by the same progress metric. It measures a different notion of
churn. Keep it separate from this initial experiment so results stay attributable.

## Software validation

```sh
python3 tests/experiments/test_prefetch_budget.py
python3 tests/experiments/test_speculative_prefetch.py
python3 tests/experiments/test_rapid_return_population.py
python3 tests/experiments/test_demand_first_eviction.py
python3 tests/experiments/test_demand_first_notifications.py
python3 tests/experiments/test_vab_classifier.py
python3 tests/experiments/test_vab_observation.py
python3 tests/experiments/test_vab_readiness.py
python3 tests/experiments/test_vab_recency.py
python3 tests/experiments/test_legacy_policy_port.py
```

These execute production policy code with mocked kernel/RM dependencies. They
cannot validate hardware VAB coverage, actual migration traffic, or a complete
Linux module build. Build against the target Linux kernel and use matching
615.71.09 user-space/GSP components for hardware evaluation.
