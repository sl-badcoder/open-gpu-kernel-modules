# Refault-distance admission experiment

Branch: `615.71.09/experiment/refault-distance-admission`.
Base: rapid-return-ratio at `7410eed3`.

This experiment uses native access-counter service and existing VAB root
selection. For eligible CPU-preferred ranges in refault mode, it explicitly
selects the accessing GPU for the pages indicated by access counters. The normal
CPU-preferred selector otherwise retains host residency. Thrashing pin/throttle
decisions and non-CPU per-page preferred-location overrides still apply.
CPU-preferred advice within an eligible range remains subject to this experiment’s
access-counter-triggered GPU migration. It controls
**additional speculative neighbors**, without filling the rest of the triggering VA block. Its default is at most one extra
2 MiB block per admitted attempt. Demand service runs first and is never denied
by this policy. This is kernel-only and needs no vLLM changes.

## Load controls

Use the usual matching UVM module and CPU-preferred/access-counter/VAB settings.
Add these parameters to `insmod` (these are module parameters, not separate modules):

```sh
uvm_perf_prefetch_refault_enable=1 \
uvm_perf_prefetch_refault_short_percent=25 \
uvm_perf_prefetch_refault_max_neighbors=1 \
uvm_perf_speculative_prefetch_sparse_enable=0
```

These are the defaults on this branch. Parameters are read-only after module
load. Reload between independent runs to reset policy history.

- `uvm_perf_prefetch_refault_enable=0`: bypass this experiment. Sparse learning
  stays off by default, so the preceding contiguous walker and rapid-return
  controller apply. Set `uvm_perf_speculative_prefetch_sparse_enable=1` explicitly to compare sparse learning.
- `uvm_perf_prefetch_refault_max_neighbors=0`: no additional speculative copies
  in refault mode. Native demand migration continues. This is different from
  disabling the experiment and returning to contiguous prefetch.
- `uvm_perf_prefetch_refault_max_neighbors=1`: default cap of one extra 2 MiB
  block. Local refault evidence can suppress that transfer completely.
- `uvm_perf_prefetch_refault_max_neighbors=4`: opt into spatial window growth
  up to four extra blocks (8 MiB). Values up to 31 allow up to 62 MiB of
  additional neighbors in the trigger's absolute 64 MiB virtual tile, before
  other limits. Growth above one neighbor is an unvalidated spatial heuristic.
- `uvm_perf_prefetch_refault_short_percent=25`: classify a return as short when
  intervening access-counter demand migration is at most 25% of GPU capacity.
  Clamped to 1–100. A **lower** value classifies fewer returns as short and makes
  this admission policy less restrictive. On a 16 GiB GPU, 25% is 4 GiB; this is
  a migration distance, not the returned block's size or a time interval.

Keep `uvm_cpu_preferred_block_promotion_enable=0` for the intended comparison.
Enabling native full-block promotion changes demand movement independently of
this policy. Existing native UVM prefetch settings also remain independent.

## Distance and partial eviction

Each VA-block GPU state stores its eviction episode and short-return streak.
There is no small GPU-wide block shadow table that loses most episodes under
oversubscription. The first successful eviction callback captures the GPU's
cumulative access-counter demand-migration byte clock. Further partial eviction
callbacks preserve that episode's start.

Before native demand service, the callback snapshots whether accessed pages
intersect that GPU state's existing per-page `evicted` mask. Only a successful
native service with that overlap can consume the episode as a return. Access to
an unrelated CPU page in the same 2 MiB virtual block does not count.

```text
return_distance = demand_byte_clock_before_this_service - eviction_byte_clock
short_return = return_distance <= GPU_capacity_bytes * short_percent / 100
```

For the intended active workload, `prefetch_refault_demand_bytes` should now
advance as counter-triggered demand pages migrate. It can legitimately stay
unchanged for stale notifications or pages held remotely by thrashing policy.
The current demand's own migration is excluded from its return distance. Byte
arithmetic uses residency bookkeeping, not copy-completion timestamps. Specifically, the clock advances by
pages that were CPU-resident and not GPU-resident before access-counter service,
and became GPU-resident afterward. Partial residency additions on a failed
service advance the clock, but the failed observation does not train the policy.
Speculative copies do not advance it. Faults, explicit CUDA migrations, peer-only
transfers, and other movement paths are outside this clock. Therefore it is a
progress proxy, not a PCIe traffic meter or exact cache reuse distance. Native
allocation retries can release/reacquire the block lock; concurrent residency
changes during that interval can bias the before/after byte estimate.

Distance remains coarse at the first block eviction episode, even though return
eligibility is checked against the page mask. Returns through a fault or API
path can leave an old episode clock and make a later return appear longer. This
can undercount short returns. New split-block GPU state starts without scalar
history; existing per-page eviction masks follow UVM's normal split logic.

## Local admission and optional growth

Policy history uses 256 fixed hash slots per GPU keyed by range lifetime ID and
absolute 64 MiB virtual tile. Hash replacement discards tile policy history only, including pressure masks and learned windows; entries
retain no VA-block/range pointers. PMM list_lock serializes policy operations.
The block episode fields also require the VA-block lock. No timers, allocation,
VAB scans, or GPU work occur under the policy lock.

A short return halves the tile's neighbor window, with a minimum of one. A
second short return of the same block suppresses the tile's neighbors entirely.
A streak separated by a full GPU capacity of demand-byte progress is forgotten.
A long return clears that block's streak and can reopen a suppressed window at
one neighbor. Separately, if at least four distinct blocks return within a tile's
observation epoch and at least half had short returns, neighbors are suppressed
for that tile. A hot block cannot pause unrelated tiles or allocations.

An epoch spans a quarter of GPU capacity in demand-byte progress (minimum
2 MiB). On the next policy event after that progress, spatial/pressure masks
reset and a suppressed window can reopen at one neighbor. An inactive tile
resets its learned size after a GPU capacity of demand-byte progress. There is
no polling, no wall-clock deadline, and no recovery merely because time passed.

With `max_neighbors>1`, four distinct remote-demand observations adjacent to
previously observed or actually speculatively populated blocks earn one size
increase. Windows double, bounded by the configured neighbor cap. Non-adjacent
distinct demand reduces pending growth credits. Duplicate offsets cannot earn
more credits within the epoch. The direction of address-sorted notifications is
not interpreted as temporal order.

Population alone earns no credit. It provides historical spatial coverage so
successful prefetching does not hide every adjacent remote notification from a
sequential stream. For example, demand for 0, prefetch of 1, then demand for 2
can contribute spatial continuation evidence. **This also fits a workload that
never uses block 1.** Coverage does not prove residency at the next observation,
cache hits, or useful consumption. That is why growth is capped at one neighbor
by default and larger caps are a separate experiment.

Candidates are visited nearest to the trigger, forward then backward at each
distance. They remain within the allocation and the absolute 64 MiB tile. Missing
VA blocks are skipped without creating backing. Copies still exclude already
GPU-resident and UVM-thrashing-protected pages. Demand's trigger is excluded from
this speculative walker entirely.

The pre-existing aggregate speculative byte cap, configured chunk size, PMA
reserve, watermark checks, no-eviction allocator retry, and final tracker waits
remain in force. VAB HOT/UNKNOWN background eviction protection is unchanged.
This experiment adds no refault-based eviction ranking or residency pinning.

## Measure the controllers separately

The older mature rapid-return controller remains an independent safety brake.
Its 20 ms age cutoff, cohort maturity, hysteresis and OFF probes are documented in
[VAB_PREFETCH_BUDGET.md](VAB_PREFETCH_BUDGET.md). In refault mode its sparse-locality
recovery gate is bypassed. Its OFF probe reserves one 2 MiB neighbor in this
mode, rather than charging 4 MiB for a trigger plus neighbor. The probe remains
single-inflight, obeys headroom and cooldown, and requires mature successful
neighbor populations before recovery; copy completion alone cannot reopen it.
At defaults, recovering the aggregate controller from OFF requires three good
batches of at least 32 successfully populated neighbors each. With one neighbor
per probe and the 60 ms cooldown, that is at least roughly 5.76 seconds of probe
opportunities, plus workload/service delays. Local suppression has its separate
byte-progress reset. Short benchmarks may therefore never recover the aggregate
window; compare the refault-only mode with backoff_enable=0 to isolate this.
Neither controller overrides demand migration.

Run the same workload with identical allocation, VAB and watermark settings:

| refault_enable | speculative_prefetch_backoff_enable | Behavior |
| --- | --- | --- |
| 0 | 0 | Fixed-cap contiguous prefetch baseline; `uvm_perf_speculative_prefetch_sparse_enable=0` |
| 0 | 1 | Contiguous prefetch with the previous rapid-return controller |
| 1 | 0 | New local refault admission with fixed aggregate cap |
| 1 | 1 | Both controllers; default on this branch |

First use max_neighbors=1 for the new modes. Then compare max_neighbors=0, 1 and
4 to separate demand-only, small speculation, and optional spatial growth.
Keep configured aggregate cap/reserve unchanged, alternate run order, and repeat
runs. Record throughput, latency, both transfer directions and bytes per token.
A shorter transfer alone is not proof of improved throughput.

Read `/proc/driver/nvidia-uvm/gpus/GPU-UUID/0/vab` before/after each workload:

- `prefetch_policy`: `refault_distance_admission` while enabled.
- `prefetch_refault_enabled`, `prefetch_refault_short_percent`,
  `prefetch_refault_max_neighbors`, `prefetch_refault_capacity_bytes`: controls.
- `prefetch_refault_demand_bytes`: this experiment's access-counter demand clock.
- `prefetch_refault_evictions`: successful eviction callbacks, including multiple
  partial callbacks; this is not an episode denominator.
- `prefetch_refault_returns`, `prefetch_refault_short_returns`,
  `prefetch_refault_last_distance_bytes`: consumed return episodes and distance.
- `prefetch_refault_plans`, `prefetch_refault_paused`: local planning queries and
  queries returning zero for local pressure, suppression or zero capacity.
  max_neighbors=0 bypasses planning and is not counted here.
- `prefetch_refault_growths`, `prefetch_refault_reductions`: actual changes to the
  local learned neighbor window. Aggregate admission can still limit the copy.
- `prefetch_refault_replacements`: tile-history hash replacements.
- Existing `prefetch_window_bytes`, `prefetch_backoffs`, `prefetch_recoveries`,
  `prefetch_reserved_bytes`, and `prefetch_capacity_stops`: aggregate controller.

The read-only module parameter `uvm_cpu_preferred_prefetched_bytes` counts actual
residency additions submitted by this walker; in refault mode all are neighbors.
It is submission/bookkeeping data, not a hardware interconnect counter. Proc
reads do not age or mutate policy. All clocks and budgets are per GPU, shared by
processes using this policy.

## Software validation

```sh
python3 tests/experiments/test_prefetch_refault.py
python3 tests/experiments/test_refault_demand_service.py
python3 tests/experiments/test_speculative_prefetch.py
python3 tests/experiments/test_demand_first_notifications.py
python3 tests/experiments/test_rapid_return_population.py
python3 tests/experiments/test_demand_first_eviction.py
python3 tests/experiments/test_prefetch_budget.py
python3 tests/experiments/test_prefetch_pattern.py
python3 tests/experiments/test_vab_classifier.py
python3 tests/experiments/test_vab_observation.py
python3 tests/experiments/test_vab_readiness.py
python3 tests/experiments/test_vab_recency.py
python3 tests/experiments/test_legacy_policy_port.py
```

These execute production policy/header, wrapper, walker and notification code
with mocked kernel/RM dependencies, including sanitizer checks. They do not
replace compiling/loading the full module against the target Linux kernel or
measuring performance on the GPU. Use matching 615.71.09 components.
