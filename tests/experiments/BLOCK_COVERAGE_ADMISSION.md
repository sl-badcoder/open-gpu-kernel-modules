# Block-coverage admission

Branch: `615.71.09/experiment/block-coverage-admission`.

This experiment changes which blocks qualify for promotion. It retains efficient
whole-trigger-block population and the bounded refill policy. The previous
page-restricted experiment reduced bytes but lost throughput; disable it here.

## Load and compare

Rebuild on Linux and keep your successful bounded-refill settings: threshold
1024, admission pressure 85%, base observations 2, replacement enabled, extra
eviction probation disabled, VAB mode 3 with your verified mapping, eager
eviction enabled, full-block promotion parameter disabled, chunk 64 MiB,
population cap 32 MiB and reserve 64 MiB. Keep refault, region migration,
backoff, recovery and sparse prefetch disabled.

Add these load-time parameters:

```bash
uvm_perf_access_counter_indicated_pages_enable=0 \
uvm_perf_access_counter_admission_coverage_enable=1 \
uvm_perf_access_counter_admission_coverage_percent=25 \
uvm_perf_access_counter_admission_sparse_observations=4
```

The new gate defaults off. Compare the same binary with coverage enable 0 and 1,
leaving indicated-pages disabled in both. Run batch 32 first and then 8 and 24,
with alternating repeated runs and separate monitoring directories:

```bash
BATCH_SIZES="8 24 32" RESULT_ROOT=results/coverage-on \
  bash ../../open-gpu-kernel-modules/tests/experiments/run_vllm_monitored.sh
```

Use `results/coverage-off` for the baseline. All gate parameters are read-only
after load. If too defensive, try sparse observations 3 before reducing the
coverage threshold; vary one parameter at a time.

## Decision

Below admission pressure, the original immediate admission remains. Under
pressure, the first observation starts probation and records its notification
page mask. Subsequent observations contribute to the union only after the
existing completed-clear and backlog fences establish fresh evidence.
Pre-clear duplicates neither advance observations nor broaden coverage.

Coverage is the number of distinct indicated OS pages divided by the actual
logical VA-block page count. A full 2 MiB block needs at least 512 KiB of
coverage at 25%; a shorter boundary block uses its own size. Repeated touches
to the same pages do not increase coverage. Hardware counter granularity limits
the precision of this evidence; coverage is not an exact application byte count.
In the current notification expansion path, backing other than a 2 MiB physical
chunk contributes only one OS page per notification; 2 MiB backing can expose
the counter's sub-granularity regions. If almost every grant is classified
sparse, this experiment effectively becomes a four-observation gate rather than
a useful discriminator. Check the broad/sparse grant split before tuning it.

Broad coverage permits admission at the normal two-observation target. Narrow
coverage requires four fresh observations, so sustained narrow demand can still
qualify without waiting indefinitely for broader coverage. If the second fresh
report broadens the union enough, the normal target applies immediately. If the
optional old eviction probation is enabled, its separately frozen target is a
floor; coverage never relaxes that target.

After admission, optional population retains its original whole-block mask,
one-shot claim, reservation and VAB eviction behavior. Coverage does not select
individual pages to copy. Native servicing and clears keep their existing paths,
and rejected optional promotions can continue accessing host memory remotely.

Probation restarts on a changed counter buffer, eviction, split, low-pressure
admission or cancelled service/clear. Stale masks are ignored and cleared on the
next probation. No global history, timer or scan is introduced. One OS-page mask
is added per allocated logical block/GPU state (64 bytes for a 2 MiB block on
a 4 KiB-page build). When enabled under pressure, fresh reports perform a mask
union and coverage decisions count mask bits. With the gate disabled, no union
or bit count is performed; state storage remains.

## Measurements

The monitor records before/after deltas and rates for:

- `uvm_cpu_preferred_admission_coverage_broad`: pressure admission grants with
  broad coverage.
- `uvm_cpu_preferred_admission_coverage_sparse`: pressure admission grants with
  narrow coverage after the higher observation target.
- `uvm_cpu_preferred_admission_coverage_deferred`: denied pressure decisions
  with narrow coverage, including duplicate notifications and the initial base
  probation. This is not a count of uniquely avoided copies.

Grants are opportunities, not guaranteed data transfers. Compare warm tokens/s,
trigger population bytes per output token, admitted population attempts and
background allocated-root releases. Do not use fewer copied bytes alone as the
success criterion. Existing trigger indicated/extra counters describe only the
notification at the eventual population, not the accumulated probation union;
outside-mask bytes can therefore be expected with whole-block copying.

Coverage is a hypothesis for block usefulness, not proof. A frequently used
narrow footprint may still benefit from early promotion. Local mocked-production
tests validate decisions and integration; Linux compilation and GPU throughput
experiments are required before drawing performance conclusions.
