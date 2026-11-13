# Threshold-triggered 4 MiB regions with VAB-ranked reclaim

Branch: `615.71.09/experiment/vab-4m-region-lru`. This is a dedicated kernel-only
placement experiment.

## Policy

- Managed allocations start CPU-preferred, with GPU AccessedBy mappings to their
  CPU backing. First touch creates backing lazily. GPU faults preserve that
  residency/remotely map it rather than force the old GPU promotion override.
- A hardware access-counter threshold notification for a remotely accessed page
  promotes the **absolute 4 MiB-aligned virtual region** containing it, clipped
  to the containing managed allocation. The triggering 2 MiB VA block is handled
  first, followed by the other half under a separate block lock.
- Move populated CPU-resident pages not already on the destination GPU; preserve
  the existing thrashing-protected page exclusion. Missing VA blocks and pages
  without CPU backing are skipped. A fully populated region can move 4 MiB; a
  sparse region or allocation boundary can move less. No new backing is created
  merely because the neighboring half lies in the region.
- Migration may allocate with eviction enabled. When free GPU capacity is
  insufficient, PMM's mandatory allocation/PMA reclaim path selects physical
  roots using VAB recency, mode 3. Free/unused/discarded roots keep priority.
- A readiness worker validates VAB ownership and migration completion. It never
  evicts data. Mandatory allocation/PMA reclaim performs eviction when needed.
- This branch contains no speculative neighborhood prefetch, adaptive budget,
  backoff, recovery, probe, or runtime mode selector. Access-counter notifications
  directly drive 4 MiB region migration.

The native hardware access counter is still configured at **2 MiB granularity**.
A notification on either half triggers software migration of its 4 MiB VA pair;
there is no assertion that hardware supplies one combined counter for 4 MiB.
The threshold counts hardware-observed remote accesses according to the device's
counter semantics, not CUDA loads or distinct bytes. Notifications can be delayed
or stale, so this is not an exact application access-count boundary.

`CPU-preferred` does not mean `cudaHostAlloc`, `cudaHostRegister`, permanent
page-locking, or eager allocation of the entire managed range. Permanently
immovable host allocations would defeat migration. The CUDA virtual pointer and
logical contents remain the same while UVM updates residency and PTEs. Explicit
application migration/advice, CPU writes, unsupported devices and other mandatory
UVM mechanisms can still affect placement; this is not an API prohibition.

## Virtual regions versus physical VAB regions

The aligned virtual unit is:

```text
base = trigger_virtual_address & ~(4 MiB - 1)
region = [base, base + 4 MiB), intersected with the managed allocation
```

It consists of up to two existing 2 MiB VA blocks. PMM still allocates and evicts
**2 MiB physical roots**. Those roots need not be physically adjacent and need
not share a VAB bit. Migration of a pair is not atomic, and residency of the first
half is not permanently pinned while the second is populated. Concurrent demand
or mandatory reclaim can change residency afterward.

The existing reverse ownership links and physical classifier remain unchanged:

```text
VAB bit = (physical_address - linear_base) / native_bit_size
native_bit_size = 64 KiB << granularity
```

With the calibrated `linear_base=0` and native bit_size=4 MiB, bit 0 covers
`[0,0x400000)`, bit 1 covers `[0x400000,0x800000)`, etc. One physical VAB bit can
cover roots belonging to different virtual regions or allocations. This branch
neither changes RM's VAB layout nor forces native granularity to 4 MiB. Check the
actual `granularity_kib` diagnostic; 4096 means 4 MiB. Keep base zero only if it is
valid for your device's mapping. The preceding calibration supports the sampled
physical region, not an automatic proof of all coverage.

## What VAB LRU means

Mandatory reclaim ranks ownership-ready full physical roots by the existing
`last_access_ns` signal associated with their physical VAB bits. The oldest usable
score is selected, even if that root is currently HOT; capacity reclaim must make
progress. VAB mode 2 is cold-first rather than LRU, so use mode 3 for this test.

This is approximate sampled recency. **Every sampled set bit refreshes its timestamp**, so a continuously set region does not become the
oldest merely because its first observed rise was long ago. Clear bits retain
their older timestamp; equal scores favor a clear candidate over a set candidate.
Two roots sharing a 4 MiB bit share its observed score.

The CURRENT dump does not document a hardware-bit reset. A sticky set bit can
therefore look recently observed even after the program stops using its data.
When all bits stay set, this policy cannot distinguish their real access order
and ties fall back to PMM list order. It is **not exact access-time LRU** and does
not manufacture access events from the samples. This experiment introduces no
hardware reset assumptions or logging reconfiguration.

Missing/stale samples, unset/incorrect coverage, incomplete ownership, split
roots, or concurrent metadata changes can prevent usable VAB ranking. Mandatory
reclaim retains baseline fallback instead of failing allocations indefinitely.
Inspect `selection_changes`, `selection_fallbacks`, and selected HOT/COLD/UNKNOWN
counts; a running program alone does not prove VAB-ranked selection occurred.

## Loading

Rebuild the kernel modules for the target Linux kernel, stop GPU clients, and
reload the matching driver stack. From the repository root, the UVM portion of a
load command is:

```sh
sudo insmod kernel-open/nvidia-uvm.ko \
    uvm_cpu_preferred_blocks_enable=1 \
    uvm_perf_access_counter_migration_enable=1 \
    uvm_perf_access_counter_threshold=256 \
    uvm_global_oversubscription=1 \
    uvm_perf_vab_mode=3 \
    uvm_perf_vab_eviction_linear_base=0
```

Keep any other device-specific driver options you actually need. CPU-preferred
placement is enabled by `uvm_cpu_preferred_blocks_enable`; there is no separate
runtime region-mode selector or legacy prefetch path on this branch.

Remove the old promotion, prefetch chunk, speculative budget/backoff/recovery,
eager-eviction, minimum-residency, and region-mode options from existing load
scripts. Their module parameters have been deleted; passing them to `insmod`
will fail with an unknown-parameter error. The command above contains the
configuration needed for this experiment.

The original VAB sample interval remains 100 ms. You can separately test
`uvm_perf_vab_interval_ms=20` to increase observation frequency; record it as an
independent experimental variable and measure its overhead.

## Observing and testing

Load-time configuration and global migration counters are exposed in:

```sh
cat /sys/module/nvidia_uvm/parameters/uvm_cpu_preferred_region_promotions
cat /sys/module/nvidia_uvm/parameters/uvm_cpu_preferred_region_migrated_bytes
cat /sys/module/nvidia_uvm/parameters/uvm_cpu_preferred_region_failures
cat /proc/driver/nvidia-uvm/gpus/GPU-REPLACE-WITH-UUID/0/vab
```

Region promotions count successful operations that moved at least some pages,
not a count of unique virtual regions or guaranteed full 4 MiB transfers. The
migrated-byte counter includes partial progress before errors; failures have a
separate counter. Stale unmapped triggering blocks are checked under lock before starting migration
and before the second half. Once a block transition starts, its own removal of
remote GPU mappings is expected; a persistent operation flag permits allocation
and mapping retries to finish without treating that unmap as an old notification. Successfully
ignored stale/no-op notifications are cleared to rearm hardware; failed operations
are not acknowledged as successful migration.

The VAB report identifies `migration_policy threshold_4MiB_VA_regions` and
`reclaim_policy allocation_driven_VAB_LRU`. Region counters are global across
GPUs/VA spaces; PMM VAB selection counters are per GPU.

For a controlled hardware test, first CPU-initialize a populated, 4 MiB-aligned
managed range. Access one half repeatedly on the GPU. After the hardware threshold
is serviced, inspect both halves' GPU residency and migrated-byte counters. Repeat
using the upper half as trigger and with an allocation boundary. Oversubscribe
with additional CPU-initialized regions and inspect VAB selection/fallback counters,
GPU/CPU residency and bidirectional migration bytes. Clear/reload between runs if
you need comparable counter baselines. Merely moving 4 MiB of virtual data does
not demonstrate that its backing occupies one physical VAB region.

For vLLM compare identical prompts, concurrency, context length and threshold,
recording throughput, per-token latency, migration traffic and fallback frequency.
The application prefix-cache hit rate measures logical reuse; this kernel
placement policy does not directly increase that metric. Its target is transfer
cost and stall time for the KV data that is actually accessed.

Run the software suite from the repository root:

```sh
python3 tests/experiments/test_region_migration.py
python3 tests/experiments/test_demand_first_notifications.py
python3 tests/experiments/test_demand_first_eviction.py
python3 tests/experiments/test_legacy_policy_port.py
python3 tests/experiments/test_vab_classifier.py
python3 tests/experiments/test_vab_observation.py
python3 tests/experiments/test_vab_readiness.py
python3 tests/experiments/test_vab_recency.py
```

Software checks include the new `test_region_migration.py`, notification routing,
CPU-preferred placement and region eligibility, VAB mandatory LRU ranking, and VAB classifier
and sampling checks. They compile extracted production C with mocked
kernel/RM dependencies and sanitizers. They do not build a complete Linux module
or prove real hardware mapping, cache-hit logging, or throughput on this machine.
