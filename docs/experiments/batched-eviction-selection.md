# Batched eviction selection experiment

This experiment changes only eager user-root eviction. Demand/PMA selection
continues to use the original full selector and its generic behavior.

Each eager batch keeps a bounded snapshot of at most 16 used roots. The
snapshot is built with a worst-first max-heap (`O(N log 16)`), then sorted
coldest-first. Protection expiry, migration score, residency age, heat, and
access epoch use the same ordering as the original selector. Current free,
unused, and discarded lists are checked on every claim, so their priority is
not cached.

Pointers in the snapshot are safe for the PMM lifetime because root chunks are
stored in the PMM root array. Candidates are not marked `in_eviction` while
being sampled. Before `chunk_start_eviction`, the list lock revalidates user
type, residency/list eligibility, and eviction state; stale candidates are
skipped. The ranking and non-force coldness decision are intentionally sampled
for one bounded selection window and refreshed only when that snapshot is
exhausted.

The existing batch limit, low-watermark accounting, pending-root draining on
errors, and prefetch exclusion are unchanged. No Linux kernel build is
required for the reference test:

```sh
python3 tests/experiments/test_batched_eviction_selection.py
```

