# Skip covered prefetch tree regions

Branch: `595.71.05/experiment/prefetch-region-skip`, based on combined
demand-first/coalesced prefetch commit `03740819`.

`compute_prefetch_mask()` now resumes iteration at the end of an already
selected tree node. The tree is immutable during mask construction, and every
page inside that node shares its ancestors, so additional walks cannot enlarge
the output. Clamp the resume position to the fault region's end because the
iterator terminates on equality. Empty selections retain normal iteration.
This changes CPU work, not migration policy, thresholds, locking or allocations.

Run `python3 tests/experiments/test_prefetch_region_skip.py`. The harness extracts
production C and compares it with the parent commit's mask loop, using production
tree/iterator macros and mocked bitmap primitives. It checks exhaustive small
bitmap combinations over thresholds 0–100, shifted and truncated trees,
20,000 deterministic randomized cases, and a fault region ending inside a
selected node. Padding before the offset is zero, as in production.

Local result: 1,930,920 mask comparisons plus the partial-region case pass with
warnings as errors, AddressSanitizer and UndefinedBehaviorSanitizer. A dense
256-page half of a 512-page tree at threshold 51 needs 10 bitmap-weight calls
instead of 2,560 (one tree walk instead of 256). This is a work-count result,
not a measured application speedup. Sparse faults and full-root selections may
see little benefit. The three existing experiment harnesses also pass.

Linux build/load and GPU workload measurements remain required; the development
host is macOS. Compare against `03740819` with identical parameters using the
[shared measurement procedure](README.md). Include sparse/random, streaming,
clustered faults, cold/warm residency and oversubscribed workloads.

Research: the [upstream prefetch implementation](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/kernel-open/nvidia-uvm/uvm_perf_prefetch.c)
uses the same per-fault tree traversal. NVIDIA's
[oversubscription study](https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/)
explains why access pattern and platform must be controlled when measuring
Unified Memory optimizations. No universally fastest parameter set is assumed.
