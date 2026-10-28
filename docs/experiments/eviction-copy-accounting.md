# Eviction copy accounting

Base: `595.71.05/experiment/demand-paced-movement` at `e91238ed`.

This branch does not change victim selection or eviction policy. It records the
payload selected for GPU-to-CPU migration for each reclaimed root after
residency and discarded-page filtering. The counters are exposed as
`uvm_perf_eviction_copy_bytes` and `uvm_perf_eviction_copy_roots`.
The root counter includes every successfully reclaimed root, including roots
with no eligible payload. The byte path measures managed-memory CPU/GPU copies;
HMM eviction paths may contribute a zero-byte root and should be excluded when
comparing managed-memory copy averages.

The byte count is the payload submitted to the CPU-copy path or GPU copy engine,
multiplied by the kernel page size. It therefore captures partial roots, unlike
assuming every reclaimed root copied 2 MiB. It is an accounting signal, not a
PCIe wire-byte measurement: compression and protocol overhead are not
represented. A migration that later fails is omitted from the completed-root
aggregate; the counters are cumulative from module load.

Compare bytes per root (`copy_bytes / copy_roots`) with D2H traffic, refaults,
and runtime for sequential, random, and mixed workloads.
