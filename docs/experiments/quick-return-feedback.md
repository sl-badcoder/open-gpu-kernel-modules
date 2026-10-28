# Quick-return eviction feedback

Base: `595.71.05/experiment/demand-paced-movement` at `e91238ed`.

This experiment adds bounded residency protection when a VA block returns to
the GPU within the existing migration return window after a GPU-to-CPU
migration. The existing migration score and residency grace remain unchanged.
The extra protection provides negative feedback without relying on decaying
access heat or touching Blackwell VAB/access-bit support.

Parameters: `uvm_perf_migration_quick_return_protection_ms` (default `250` ms
per return) and `uvm_perf_migration_quick_return_max_protection_ms` (default
`2000` ms). Set the per-return value to zero to disable this experiment.
The marker is armed only when the migration cause is
`UVM_MAKE_RESIDENT_CAUSE_EVICTION`; ordinary CPU demotion does not count.
The quick-return streak decays with the existing migration-history decay
interval. Measure quick returns, eviction/refault bytes and runtime; this does
not observe GPU-local accesses.
