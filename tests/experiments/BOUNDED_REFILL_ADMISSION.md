# Bounded refill with return-distance measurements

Branch: `615.71.09/experiment/bounded-refill-admission`.

This experiment replaces continuous watermark-driven background reclaim with a
small refill requested by failed optional population admission. It retains the
original two-observation pressure admission and measures eviction-return distance
without increasing probation by default. It does not add the earlier turnover
controller or force access-counter notifications to migrate pages.

## Load and compare

Rebuild the module on Linux, then add these explicit parameters to your loader:

```bash
uvm_perf_access_counter_admission_enable=1 \
uvm_perf_access_counter_admission_pressure_percent=85 \
uvm_perf_access_counter_admission_observations=2 \
uvm_perf_access_counter_admission_eviction_enable=0 \
uvm_perf_access_counter_admission_distance_stats_enable=1 \
uvm_perf_prefetch_replacement_enable=1 \
uvm_perf_eager_eviction_enable=1 \
uvm_perf_eager_eviction_batch_count=16 \
uvm_perf_speculative_prefetch_limit_mb=32 \
uvm_perf_speculative_prefetch_reserve_mb=64
```

Retain threshold 1024, VAB mode 3, linear base 0 (with your verified mapping),
CPU-preferred blocks enabled, full-block promotion disabled, chunk size 64 MiB,
and refault/backoff/recovery/sparse/region-migration disabled. Retain your eager
high/low 99/90 and interval 25 ms for the comparison, although replacement mode
does not drain toward the low watermark. Admission pressure still uses PMA
occupancy, independently of reclaim's byte target.

The new replacement flag defaults **false**. For a matched comparison, change
only `uvm_perf_prefetch_replacement_enable` between 0 and 1 on this binary.
Keep extra eviction probation disabled and distance statistics enabled in both.
An additional leg can enable extra probation, but that tests two mechanisms
together. All parameters are load-time, read-only; reload between configurations.

Run batches 8, 24 and 32, preferably multiple alternating off/on runs. The small
batch matters because eliminating background eviction entirely previously hurt
useful promotions. Example from the vLLM fork:

```bash
BATCH_SIZES="8 24 32" RESULT_ROOT=results/bounded-refill-on \
  bash ../../open-gpu-kernel-modules/tests/experiments/run_vllm_monitored.sh
```

## Refill behavior

- Optional population retains the strict in-flight byte cap and demand reserve.
  Usable headroom includes PMA-free bytes and reusable whole USER roots in PMM's
  free lists. NUMA roots are conservatively excluded from that extra credit.
- A capacity denial requests a bounded refill. Concurrent denials coalesce into
  one pending root count; they do not build an unbounded work queue. A denial
  caused solely by the in-flight cap does not request replacement.
  The pending count retains the full shortfall: an empty 64 MiB reserve plus
  a 32 MiB refill requests 48 roots, serviced in batches of at most 16.
- The asynchronous worker rechecks available headroom and waits for admitted
  optional populations to finish before reclaiming. New optional population is
  blocked during active reclaim. There is no wait for eviction in the counter
  service path; native counter servicing and clearing continue.
- The target is the demand reserve plus a small refill window, bounded by the
  configured population cap, batch count and 32 MiB. With the settings above it
  is 64+32 MiB. The worker does not continue reclaiming toward 90% occupancy.
- Reclaim uses the existing VAB cold-root selection and recency order. HOT or
  UNKNOWN used roots remain protected from optional background reclaim. Already
  reusable PMM free roots are credited as headroom rather than reclaimed again.
  Native mandatory allocation/PMA eviction remains available through its existing
  path, with its original fallback behavior.

A refill is an opportunity for subsequent notifications to retry, not a promise
that any particular denied block will be populated. If there are no safe cold
candidates, optional population can remain remote. The workqueue interval still
affects retries after an unsuccessful scan; this is not a hardware-bandwidth
sampling controller. The first pending request expedites work, productive full
batches with remaining shortage continue immediately through the workqueue, and
zero-progress scans use the normal interval. Duplicate denied notifications do
not repeatedly expedite unsuccessful scans.

## Return-distance histogram

Distance is **CPU-policy population bytes on the same GPU partition** between a
tracked complete logical block eviction and a return. It is neither elapsed time
nor PCIe wire bytes. Partial evictions and discard-only removals do not start
history. Native returns invalidate old history before this policy uses it.
History follows block/GPU lifetime and splits; no global lookup table or scan is
introduced. The distance clock advances only on nonempty successful CPU-policy
resident transitions, even when extra probation is disabled.

Twenty cumulative module counters use this name format:

`uvm_cpu_preferred_return_<bucket>_<metric>`

Buckets are `lt64m`, `64m_256m`, `256m_1g`, `1g_4g`, and `ge4g`, with inclusive
lower and exclusive upper bounds. Metrics are:

- `probations`: pressure-probation starts while complete-eviction history exists;
  duplicates within one probation do not increment this. A buffer change or
  cancelled probation can start another, so it is not a unique-block count.
- `grants`: admission opportunities with return history, classified at grant
  time, including low-pressure grants. A grant can fail to transfer data.
- `populations`: first nonempty successful CPU-policy population after a tracked
  complete eviction, counted once. This is separate from admission opportunities.
- `population_bytes`: actual changed-residency bytes in those populations.

The three stages may land in different distance buckets as other blocks move.
Do not divide grants by probations within a bucket and call it a success rate.
Population occurs after admission, and may include a neighbor below pressure.
Later partial refills are outside the completed-eviction episode measurement.

The updated monitor records rates and puts the actual-population histogram and
byte fractions in `summary.json` under `return_distance_histogram`. Use before/
after deltas; counters are driver-wide. For older modules or missing/reset
counter coverage, that summary is null. Zero measured return bytes produce null
fractions rather than a false zero return rate.

Statistics default enabled independently of extra probation. To measure their
overhead, disable both `uvm_perf_access_counter_admission_distance_stats_enable`
and `uvm_perf_access_counter_admission_eviction_enable` in a separate matched run.

Compare warm tokens/s alongside trigger population bytes per output token,
background releases, replacement requests, and the return-distance byte
distribution. A reduction in copies accompanied by slower inference can mean
remote-access costs outweigh the copies avoided. Source tests cannot establish
GPU performance or prove PCIe saturation.

The VAB proc snapshot also exposes `replacement_pending_roots`,
`replacement_cached_free_bytes`, `replacement_usable_free_bytes`, cumulative
`replacement_requests` and `replacement_freed_roots`, and splits background
releases into `background_cached_root_releases` and
`background_allocated_root_releases`. A cached release is allocator bookkeeping,
not migration of live data. An allocated release also need not be a full 2 MiB
data transfer. The monitor treats pending roots and free bytes as gauges, and
requests/releases as counters. If repeated requests grow while freed roots stay
zero, inspect cold-candidate availability, VAB validity and headroom before
concluding the capacity controller is effective.
`replacement_requests` counts shortage-denied admission attempts, including
duplicates while a request is pending; it is not a count of workqueue kicks.

Local validation compiles production policy and PMM routines with mocked hardware
and exercises the monitor. Full Linux kernel compilation and GPU tests remain
required on the target system.

## Cheaper denied population attempts

The follow-up optimization allocates a migration context only after a byte
reservation succeeds. Denied, empty-plan and duplicate claimed attempts avoid
context allocation/free; successful population retains the same size, order,
byte limits and eviction policy. No additional module parameter is required.

If context allocation fails after admission, normal completion releases the
reservation before reopening the trigger for another notification. This counts
as an admitted failed attempt. In the old optional backoff/recovery mode a failed
recovery probe can therefore defer the next probe; keep that controller disabled
for this bounded-refill experiment as specified above. GPU performance impact of
the allocation-order change still requires a matched hardware comparison.
