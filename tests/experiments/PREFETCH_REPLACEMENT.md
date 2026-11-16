# Request-driven replacement experiment

This experiment targets CPU-preferred access-counter population. It preserves
GPU remote access to host memory, the no-eviction population allocator, and VAB
cold-only background selection. It does not force demand promotion or change
physical VAB mapping.

## Loading and comparisons

Rebuild and reload `nvidia-uvm.ko`. Keep your existing access-counter threshold,
admission percentage, VAB mode/mapping, speculative byte cap and reserve fixed.
For the new behavior, add these arguments to your existing `insmod` command:

```bash
uvm_perf_eager_eviction_enable=1 \
uvm_perf_prefetch_replacement_enable=1 \
uvm_perf_prefetch_turnover_enable=1
```

For the matched old-policy baseline, use:

```bash
uvm_perf_eager_eviction_enable=1 \
uvm_perf_prefetch_replacement_enable=0 \
uvm_perf_prefetch_turnover_enable=1
```

Replacement defaults OFF, turnover accounting defaults ON. These are load-time
parameters. Setting both new flags to zero skips turnover accounting, although
per-block history storage remains part of the compiled GPU state. Replacement
implies accounting even if the accounting flag is zero. Setting eager eviction
to zero also disables request-driven reclaim and repeats the fill-and-stop
experiment; it is not the recommended experimental setting.

Use the same batch-8 and batch-32 workload, threshold 1024 and admission pressure
85 for the first comparison. Alternate policies across repeated runs. Keep
refault, sparse learning, legacy speculative backoff and recovery disabled.
The replacement policy has its own byte-based hysteresis described below.

## Replacement behavior

A successfully admitted access-counter callback first tries the existing bounded
population reservation. If it fails because there is less than the configured
reserve plus outstanding reservations plus one 2 MiB population quantum, it
records a coalesced reclaim request. In replacement mode, usable free bytes
include PMA-free memory plus whole USER roots cached by PMM. The cache count is
maintained under the existing list lock, with no free-list scan per notification.
NUMA roots are not credited because external page references can prevent reuse.
Failure of the speculative inflight limit
alone never requests eviction. Requests retain no VA-block pointer.

The worker rechecks current headroom. It reclaims at most the space needed for
a bounded refill window above the reserve: the lesser of the configured
speculative byte limit and eviction batch, with a hard 16-root limit (32 MiB
with the default settings). This avoids a worker round trip per 2 MiB copy.
Outstanding reservations are still subtracted and the inflight byte cap remains
unchanged. It uses the existing VAB-cold picker and
minimum-residency protection. Request-driven reclaim skips already-free cached
roots: returning them to PMA does not increase combined usable headroom. Native
mandatory eviction and legacy eager recovery retain cache draining. It does not
drain to the low watermark. Population
submissions and background reclaim remain mutually exclusive. Native mandatory
PMA/demand eviction remains unchanged.

Multiple notifications coalesce into a bounded pending root count, not an
unbounded queue. Once headroom is present, pending reclaim is discarded. The
worker uses the existing scheduling/readiness polling; there is no synchronous
eviction inside access-counter service.

## Turnover metric and control

Every successful CPU-policy population advances a per-GPU byte clock using
`pages_changed_residency`. Both trigger and neighbor movement advance this clock.
Logical per-block/GPU masks track populated live pages and evicted pages. An
explicit managed eviction records the byte clock and GPU residency relinquished;
discarded/unbacked pages are excluded. A later population counts only the
intersection with outstanding evicted pages as repopulation, clearing those bits
once counted. Neighbor population does not feed the trigger controller.

Distance is intervening population bytes, not milliseconds. Short trigger-return
bytes are bucketed at one-quarter, one-half and one full current tracked resident
budget. The budget is CPU-policy-populated live residency tracked by these hooks,
with a 64 MiB minimum normalization. It is not total logical KV size or a complete
inventory of native UVM residency. Other native migration/discard paths can make
this gauge approximate. GPU-state teardown removes live accounting; VA-block
splits preserve both history masks and their timestamp.

A single oldest outstanding eviction stamp is kept per block, rather than a
stamp per page. Later partial evictions can therefore have their distance
conservatively overestimated. This can miss short returns; it does not falsely
shorten them. Histories disappear when the logical block/GPU state is destroyed.
Accounting follows residency bookkeeping after successful submission, not DMA
completion or measured PCIe wire bytes. Failed operations are not reconstructed.

After at least `max(64 MiB, tracked resident budget / 16)` trigger-population bytes:

* At least 50% returning within one-quarter of the budget enables throttling.
* Below 25% restores normal request-driven reclaim.
* Between these thresholds preserves the current mode.

In throttled mode, reclaim is limited to approximately 1/64 of the tracked live
budget per existing valid VAB sample epoch, rounded up to whole physical roots,
with at least one root allowed. Repeated worker calls in the same epoch cannot
refill the allowance. This preserves probes for changed workloads. Normal mode
has no additional epoch quota. The existing VAB interval consequently affects
the throttled reclaim rate; the experiment does not eliminate hardware/workload
sensitivity, and the fractions are initial heuristics to validate.

The `turnover_request_stride` gauge is a mode value: 0/1 means normal,
4 means throttled. It does not sample one out of four notifications.

## Monitoring

The existing `run_vllm_monitored.sh` and updated `monitor_uvm.py` automatically
capture new `/proc/driver/nvidia-uvm/gpus/GPU-.../0/vab` fields. No additional
monitor command is needed. Important values:

* `replacement_enabled`, `turnover_enabled`: effective experiment settings.
* `replacement_cached_free_bytes`: usable whole USER roots in PMM's cache.
* `replacement_usable_free_bytes`: headroom source before reserve/reservation
  subtraction (combined PMA/cache bytes in replacement mode, PMA-only in baseline).
* `background_cached_root_releases`, `background_allocated_root_releases`:
  successful background releases selected from already-free cache versus allocated
  roots. Allocated roots can include unused VA backing; use eviction-byte counters
  for the managed-residency scope. Replacement mode should not release cached roots.
* `turnover_population_bytes`, `turnover_trigger_population_bytes`,
  `turnover_repopulated_bytes`, `turnover_evicted_bytes`: cumulative residency
  bookkeeping, not PCIe bytes. Repopulation total includes neighbors.
* `turnover_short_quarter_bytes`, `turnover_short_half_bytes`,
  `turnover_short_budget_bytes`: cumulative trigger-only short-return bytes.
* `turnover_resident_bytes`: current tracked live budget gauge.
* `turnover_last_population_bytes`, `turnover_last_short_quarter_bytes`: the last
  evaluated trigger observation; their ratio is the most recent control input.
* `turnover_window_*`: current incomplete observation, not the last decision.
* `turnover_request_stride`, `turnover_epoch_roots_limit`,
  `turnover_epoch_roots_used`, `turnover_epoch_pauses`: controller mode/allowance.
* `replacement_requests`, `replacement_pending_roots`,
  `replacement_freed_roots`: shortage attempts, coalesced pending gauge and actual
  successful request-driven root frees.
* `replacement_selection_ns`: cumulative elapsed selection time, including time
  inside selection locks, excluding acquisition of the outer PMM mutex.
* `replacement_drain_ns`: elapsed root-free/drain time, including tracker waits
  and other release work; it is not pure DMA latency.
* `replacement_accessed_roots`, `replacement_unknown_recency_roots`: background
  roots with physical VAB recency after readiness, or without known recency.
  A 4 MiB bit can cover multiple roots. This is coarse physical-use evidence,
  not the number of logical pages used. Counts include the legacy baseline.

Timing/recency fields cover background eviction in both modes. New state and
pending allowance values are gauges; the monitor gives rates only to cumulative
counters. Summary derived ratios include repopulation/population and
short-quarter/trigger-population fractions. Use deltas rather than raw cumulative
values; one loaded module can serve multiple runs. Check per-pass intervals when
interpreting warm throughput separately from cold initialization.

Success means retaining batch-8 throughput while reducing batch-32 movement and
improving throughput, not merely reducing bytes. Compare cold and warm passes
separately. No hardware performance result is claimed before these tests.

## Local checks

```bash
python3 tests/experiments/test_prefetch_replacement.py
python3 tests/experiments/test_replacement_integration.py
python3 tests/experiments/test_replacement_worker.py
python3 tests/experiments/test_replacement_requests.py
python3 tests/experiments/test_pmm_cached_roots.py
python3 tests/experiments/test_demand_first_eviction.py
python3 tests/experiments/test_monitor_uvm.py
```

These compile extracted production policy/accounting/worker functions with
ASan/UBSan and test monitor classification. A full Linux module build and the
GPU experiments are still required on the target machine.
