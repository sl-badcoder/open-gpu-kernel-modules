# Cold-only background eviction with emergency fallback

Branch: `595.71.05/experiment/cold-only-background-eviction`.
Parent: `595.71.05/experiment/prefetch-region-skip` (`12ceb155`).
This includes the prefetch traversal optimization and its combined
demand-first/coalesced baseline.

## Policy

New load-time parameter: `uvm_perf_eager_eviction_cold_threshold=0`.
It is an inclusive limit on **decayed remote access heat**, not the hardware
notification threshold. Zero admits only zero-heat used roots to background
eviction. Positive values admit roots with heat at or below that value.
The existing 25 ms heat-aging interval and all other numeric defaults remain.

Background selection preserves free (nonzero then zero), unused and discarded
root priority. Used roots must meet the heat cutoff and have expired migration
grace when migration-aware eviction is enabled. Among eligible used roots,
choose lower heat, lower migration score, older residency, then older sampled
access. Equal candidates retain list order. The unit is a physical 2 MiB root,
not an independently ranked 4 KiB page.

High-watermark recovery still obeys the cutoff and migration grace. Below high
water, the existing hot/cold contrast or zero-heat staleness condition also
applies. A second candidate tracked in the same scan handles all-zero sets:
a fresh low-score root cannot hide a stale higher-score root.

When no eligible root exists, background eviction stops its current batch,
drains already-submitted evictions and retries on the existing aging timer.
If watermark recovery remains active, speculative prefetch stays paused.
Ordinary allocation-driven eviction and PMA reclaim remain unchanged and may
evict hot/protected used roots when required. No roots are permanently pinned.
This retains the existing fallback; it does not guarantee every allocation
succeeds when other constraints, such as pinned memory, prevent reclaim.

## Reduced selection work and corrected victim admission

Previously, a full used-list scan could report that a cold candidate existed,
then a separate selector could choose a hotter root based on migration score
and residency age. The new background picker computes global hottest heat,
the eligible victim and stale-zero fallback in one used-list scan. It claims
that same root under `list_lock`, with the existing PMM mutex held. There is no
unlocked candidate snapshot or second selection scan. Complexity remains O(N)
per used victim; this is not the separate batched-selection experiment.

## Limits and workload coverage

Access counters report accesses to memory on other processors; ongoing local
VRAM accesses do not refresh this score. A locally busy root can therefore
decay to zero and become eligible after its grace period. Migration history
protects recent/repeated returns, but neither signal identifies all local
reuse. See NVIDIA's [current Unified Memory guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#access-counter-migration).

No eviction policy is fastest for every access pattern. Compare this branch
against `12ceb155` on the target GPU using the [shared measurement procedure](README.md):

- Sequential/streaming: test whether cold-only recovery delays useful lookahead.
- Uniform random and sparse: measure transferred bytes as well as runtime;
  a low sampled score is not proof that data is unused.
- Skewed random/hot working set: check retention, faults and remote traffic.
- Repeated CPU/GPU reuse and phase changes: check migration grace, refaults and
  how quickly formerly hot roots become reclaimable.
- All-hot oversubscription: verify background refusal, speculative backpressure
  and allocation fallback, including concurrent non-UVM VRAM allocations.

Use matching 595.71.05 userspace/GSP, a release build, identical loaded
parameters, cold/warm residency runs and at least five timed repetitions.
Start with cutoff zero; only then sweep positive cutoffs separately from the
hardware access-counter threshold. Changing this parameter requires a module
reload. Building or fetching does not change the running driver.

## Validation

`python3 tests/experiments/test_cold_only_background_eviction.py` extracts actual
production background, demand and batch functions. With mocked lists, locks,
clock and GPU/PMA operations, it reproduces the parent's admission mismatch,
then checks cutoff boundaries (including 64-bit heat), grace expiry, stale
selection, priorities, atomic claiming, allocation fallback, failed migration,
short-batch draining and low-water targets. The two-used-root admission example
visits four list entries in the parent versus two in this branch.

The test uses warnings as errors plus AddressSanitizer/UndefinedBehaviorSanitizer.
It does not validate real concurrency, GPU DMA, error recovery inside external
migration functions, or actual Linux APIs. A Linux module build/load, lockdep
stress and GPU workload measurements remain required on a suitable machine;
the development host is macOS. No application speedup has been measured here.
