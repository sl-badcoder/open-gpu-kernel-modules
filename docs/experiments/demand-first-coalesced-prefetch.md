# Combined demand-first and coalesced-prefetch experiment

Branch: `595.71.05/experiment/demand-first-coalesced-prefetch`.

This version merges `595.71.05/experiment/demand-first-notifications`
(`0f63efa9`) and `595.71.05/experiment/coalesced-prefetch` (`bddc3046`).
Both source experiments share baseline `f735561b`. Neither batched-PMA-free nor
batched-victim-selection changes are included. Numeric module defaults and the
rejected `2f4de8ef` optimization are unchanged/absent respectively.

Demand and notification flags are handled before up to eight speculative
triggers are dispatched. The queue drains at its bound or before the existing
GPU/VA-space locking context changes. Dispatched triggers run through the
existing range-local coalescing policy, with at most eight advertised windows.
Covered work stays retryable; coverage ends after the final GPU tracker wait.
The eager-eviction admission remains held through that wait.

Queue entries drain serially within one service thread. Coalescing applies to
overlap with other concurrently active calls; this version does not add
persistent completed-window caching or an asynchronous prefetch worker. Gains
from the two experiments are not guaranteed to add together.

## Validation

Run all three tests from the repository root:

```sh
python3 tests/experiments/test_demand_first_notifications.py
python3 tests/experiments/test_coalesced_prefetch.py
python3 tests/experiments/test_demand_first_coalesced_prefetch.py
```

The integration harness compiles the actual production queue helpers and
coalesced policy together, mocking only external kernel/GPU APIs. It tests
queue deferral, bounded flushes, duplicate triggers, partial/full overlap,
retry publication, GPU isolation, error discard and completion cleanup.
All harnesses use compiler warnings as errors plus ASan/UBSan. This is not a
Linux kernel build, true parallel stress test, or GPU performance validation.

Build and load this branch with the same parameter profile used for your two
successful individual tests. Fetching/building alone does not replace the
running module. Compare baseline, each individual branch and the combined
branch under the same workload, residency setup and build mode. See the
[shared comparison guide](README.md) for build and timing instructions.
