# Coalesced in-flight prefetch experiment

Base: `f735561b`. This branch alone coalesces concurrently active, overlapping
prefetch windows in the same managed range and GPU. Each call snapshots older
windows, advertises its own window in an eight-slot range-local table, and skips
covered block indices before locking or migrating them. The table lock is held
only while copying the intervals or publishing/removing a slot.

Coverage lasts through the existing final GPU completion wait. There is no
persistent completed-window cache, so eviction and later CPU accesses cannot
leave stale coverage. Slots are bounded; a full table makes new callers proceed
without advertising another slot. GPU identity separates windows on different
GPUs. VA-space read locking keeps the range and GPU registration alive.

An older owner may stop before copying every advertised page. Therefore skipped
attempts publish their retry bit under the range lock during the snapshot
handoff, before older owners can remove coverage, and remain notification-retryable; coalescing never permanently marks those
pages complete. This inherits the baseline's notification-driven retry behavior;
it does not add a timer or guarantee an immediate retry. There is no benefit for
serial, non-overlapping signals, and interval bookkeeping can cost more than it
saves in that workload. Benchmark this branch independently.

Run `python3 tests/experiments/test_coalesced_prefetch.py`. The extracted-C test
uses ASan/UBSan, covers the actual interval helpers and orchestration, and mocks
GPU migration. A Linux module build and GPU test remain required.
