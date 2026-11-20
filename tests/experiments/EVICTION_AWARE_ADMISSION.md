# Eviction-aware admission experiment

Branch: `615.71.09/experiment/eviction-aware-admission`.

This experiment leaves first-time regions at the original two clear-separated
observations under pressure. A recently completely evicted logical VA block/GPU
requires four. Below admission pressure, admission stays immediate.

Add these parameters to your existing loader, keeping threshold 1024, pressure
85%, VAB mode 3, eager eviction and the other successful baseline settings:

```bash
uvm_perf_access_counter_admission_enable=1 \
uvm_perf_access_counter_admission_pressure_percent=85 \
uvm_perf_access_counter_admission_observations=2 \
uvm_perf_access_counter_admission_eviction_enable=1 \
uvm_perf_access_counter_admission_eviction_mb=256 \
uvm_perf_access_counter_admission_eviction_extra=2
```

Rebuild and reload `nvidia-uvm.ko`. For the original two-observation comparison on
this same binary, change only `uvm_perf_access_counter_admission_eviction_enable=0`.
The branch defaults to base 2 and eviction probation enabled; the general pressure
default remains 95%, so explicitly retain your tested 85% setting. Parameters are
load-time, read-only. Keep refault, sparse, backoff and recovery disabled.

## Decision and overhead

After successful eviction removes the last GPU-resident page of a CPU-policy
block, save a per-GPU population-byte clock in that logical block/GPU's history.
Partial evictions and discard-only removals do not start an episode. Each successful CPU-policy population
advances the clock by its actual changed-residency bytes, including trigger and
neighbor pages. These are driver residency-accounting bytes, not PCIe wire bytes.
There is no timer, global history lookup, eviction scan or application knowledge.

At the beginning of pressure probation, an eviction distance strictly below
256 MiB selects base+extra observations, capped at eight. Freeze this choice until
grant or probation cancellation, even if other migrations cross the distance
boundary meanwhile. Existing clear completion and notification-backlog fences
still apply. A changed buffer restarts probation. Low pressure, eviction and
service/clear failures cancel probation as before.

Actual CPU-policy population consumes the eviction episode once. Native returns
are detected lazily from nonempty GPU residency before admission or prefetch and
invalidate the old episode; a subsequent complete eviction records a new stamp.
Splitting a VA block copies history to both halves and restarts their probation. Empty attempts and failed
make-resident calls preserve it. Mapping errors after a successful resident
transition still account the resident bytes. Region history disappears with its
VA-block/GPU state; the GPU clock is monotonic for its lifetime and needs no
resident-byte teardown subtraction. Idle time does not age history; only population
distance does. Native non-CPU-policy migrations are outside this clock.

Cost: one atomic add per nonempty CPU-policy population, one clock read per new
pressure probation, and one read when recording complete eviction. Adds 16 bytes
per allocated block/GPU history plus eight bytes per GPU; the frozen observation
target uses existing admission padding. Disabling the experiment skips population
and eviction tracking, while retaining the existing admission gate.

## Compare

Run matched batches 8, 24 and 32 with eviction probation off/on. Keep external
VRAM reservation, hardware threshold and eager settings fixed. Compare warm
tokens/s, trigger/neighbor population bytes per output token and background
evictions, alongside these cumulative sysfs counters (use before/after deltas):

- `uvm_cpu_preferred_admission_eviction_probations`: extra-probation starts;
  buffer changes can count another start for the same block.
- `uvm_cpu_preferred_admission_eviction_deferred`: decisions denied while the
  frozen target exceeds the base. Includes the base probation observations;
  it is not a count of uniquely avoided transfers.
- `uvm_cpu_preferred_admission_return_bytes`: first CPU-policy population bytes
  after a tracked complete eviction, regardless of distance or admission pressure.
  Later partial refills are not included.

The monitor records these counters and their rates. Existing total request,
deferred, repeated and low-pressure counters keep their meanings.

If performance becomes too defensive, first try extra=1 or a smaller distance
budget such as 64 MiB; zero extra or zero distance removes extra probation.
A short return is evidence of replacement churn, not proof that migration was
unhelpful. Remote accesses can cost more than the avoided copies, so this is a
hardware experiment rather than a guaranteed throughput improvement. VAB eviction
ordering and prefetch byte limits are unchanged.

Local tests exercise production policy, admission gate, eviction hook and
population accounting with mocked hardware and ASan/UBSan. Linux module build and
GPU throughput validation must run on the target machine.
