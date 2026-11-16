# Counter admission experiment

Branch: `615.71.09/experiment/counter-admission`.

This experiment restores the ordinary CPU-preferred residency choice and controls
the optional access-counter-triggered CPU-policy migration before it runs. It does
not force the notification's accessed pages into GPU residency to advance a
refault clock. Hardware access-counter thresholds and VAB mapping are unchanged.

## Behavior

- Admission is enabled by default. Below 95% PMA usage, the first eligible
  notification may trigger the existing range-prefetch path, subject to its fixed
  byte cap, reserve, allocator checks and eager-eviction watermarks.
- At or above 95%, or while eager eviction is active/recovering, the first
  observation of a virtual 2 MiB block/GPU is deferred.
  The pages remain remotely accessible on the CPU. The notification is still
  cleared through the normal managed-memory clear path.
- Admission requires another observation of that block through the same buffer
  after the first observation's clear has completed and the old queued ring
  backlog has been consumed. Repeated notifications in one batch or later batches
  before that boundary cannot satisfy the repeat. This is software region-level
  reaccess evidence, not proof of a second threshold crossing by the same page or
  the same hardware counter entry.
  Capturing PUT after completion also excludes the report which wakes servicing
  if it is already queued: admission can require a third or later threshold
  notification. This conservative fence deliberately favors correctness over
  counting every potentially useful report.
- Under pressure, an admitted range-prefetch attempt is limited to its trigger
  block (up to 2 MiB of eligible CPU-backed pages); neighbors are excluded.
  If pressure begins during a larger attempt, stop before the next neighbor.
- Low pressure clears pending admission history; eviction also clears it. State
  follows the VA-block/GPU lifetime, not recycled physical roots. There is no
  wall-clock sampling window, global history hash or application-specific logic.
- Admission grants are opportunities, not guaranteed transfers: existing
  watermarks, resident-page filtering, thrashing safeguards and failed allocations
  can still prevent movement. A denied first observation does not change the
  native residency selector for non-CPU preferred pages or HMM.

The optional full-block promotion feature, if explicitly enabled, is gated too.
Its default is now zero. The legacy refault flag defaults false and is ineffective
whenever admission is enabled, even if a previous reload script passes it as one.
For clarity pass it as zero in both sides of the experiment. Old backoff/recovery
also default false; sparse learning remains false. Their code is retained for
separate comparisons, not deleted.

Replayable-fault full-block promotion is a separate feature and is not controlled
by this access-counter admission gate. Keep `uvm_cpu_preferred_block_promotion_enable=0`
in both legs, as in your existing loader. Otherwise a fault can still populate a
whole block through that separate path.

## Implementation and cost

- `uvm_cpu_block_policy_admit_counter()` in `uvm_cpu_block_policy.c` makes one
  decision per eligible virtual block service, under the VA-block lock.
- `uvm_counter_admission.h` contains the region decision and buffer fence policy.
  Region state is 16 bytes with normal alignment, in existing per-block/GPU state.
  If state must be allocated and allocation fails, defer instead of promoting.
- `service_notifications_in_block()` in `uvm_gpu_access_counters.c` gates both
  full-block promotion and queued range prefetch. Deferral does not skip clearing.
  Failed service/clear cancels the pending observation.
- `counter_admission_prepare_fetch()` nonblockingly tests the existing clear
  tracker and captures PUT **after** completion, then the existing fetcher drains
  the captured number of entries in GET order. A batch spanning that boundary is
  conservatively treated as old. No counter-clear wait, ring flush or hardware
  reconfiguration is introduced for normal admission. Tracker polling begins only
  after a buffer first defers an observation; it then persists for its lifetime.
- PMA usage is the existing driver allocator's usage measurement, not the vLLM
  logical KV occupancy. It includes pressure from other device allocations.
  With unavailable PMA statistics it currently follows the existing usage helper's
  zero-usage fallback, allowing baseline behavior.

The 95% gate is a hard occupancy threshold, with no separate high/low hysteresis.
Once eager recovery has been latched by its own high watermark, its active/recovery
state also holds the gate in pressure mode until that recovery ends. A workload
oscillating around 95% without reaching the eager high watermark can still switch
between modes; measure that behavior before adding another stateful controller.

There is no permanent-denial latch, but reaccess is not guaranteed: another report
must arrive after the clear/fence. Continued pressure, failed capacity admission or
no subsequent remote reports can leave pages on the CPU. Reports through another
buffer restart that region's probation. This deliberately conservative experiment
must be evaluated for increased remote traffic as well as reduced copy traffic.

## Load and compare

Keep the existing CPU-preferred, access-counter, VAB and eviction parameters.
Use these explicit settings for both configurations:

```bash
uvm_cpu_preferred_block_promotion_enable=0 \
uvm_cpu_preferred_prefetch_chunk_mb=64 \
uvm_perf_prefetch_refault_enable=0 \
uvm_perf_speculative_prefetch_backoff_enable=0 \
uvm_perf_speculative_prefetch_recovery_enable=0 \
uvm_perf_speculative_prefetch_sparse_enable=0 \
uvm_perf_access_counter_admission_pressure_percent=95
```

For the baseline add `uvm_perf_access_counter_admission_enable=0`; for the
experiment use `=1`. These are parameters to the existing `nvidia-uvm.ko`, not an
additional module. Reload between configurations; the new parameters are read-only.
Do not compare against the forced-demand refault path and call the difference a
measurement of admission bookkeeping overhead. The admission-on comparison also
includes the pressure-specific trigger-only cap; it is not a gate-only ablation.

First use batch 24 as before; then repeat at a smaller batch where prefetch pays
off. Keep the external VRAM reserve, host workload, fixed byte budget, eager
watermarks and access threshold identical. The monitored wrapper saves all these
settings automatically. Source-level tests cannot validate the ordering guarantees
of the actual GPU notification/clear machinery; confirm deferred and repeated
counts progress and inspect throughput on hardware before adopting this policy.

Example result labels when running from the vLLM fork:

```bash
RESULT_ROOT=results/admission-off bash ../../open-gpu-kernel-modules/tests/experiments/run_vllm_monitored.sh
# Reload the module with admission=1, then:
RESULT_ROOT=results/admission-on bash ../../open-gpu-kernel-modules/tests/experiments/run_vllm_monitored.sh
```

## Measurements

The monitor recognizes these cumulative sysfs counters under
`/sys/module/nvidia_uvm/parameters/`:

- `uvm_cpu_preferred_admission_requests`: eligible block-service decisions.
- `uvm_cpu_preferred_admission_deferred`: denied decisions (including state failures).
- `uvm_cpu_preferred_admission_low_pressure`: immediate grants below the pressure threshold.
- `uvm_cpu_preferred_admission_repeated`: grants backed by a post-clear repeat.
- `uvm_cpu_preferred_admission_state_failures`: failures to obtain per-GPU block state.

The first four satisfy `requests = deferred + low_pressure + repeated` over a
quiescent snapshot interval. They are decisions, not unique regions or successful
copies. Use differences, not lifetime totals. `uvm_cpu_preferred_prefetched_bytes`
now includes both the trigger and neighbors in the legacy CPU-policy path; do not
label all of it speculative bytes. Keep checking PCIe RX/TX and completed tokens.
`uvm_cpu_preferred_trigger_prefetched_bytes` and
`uvm_cpu_preferred_neighbor_prefetched_bytes` expose that split directly. Their
quiescent sum equals the total; the monitor captures both, including delta rates.
Trigger-block bytes can include pages beyond those accessed by the notification.
Admission decision counters remain zero when admission is off; the split byte
counters still record CPU-policy prefetch. With refault disabled its demand-distance byte
counter is no longer a measure of this path.

If immediate grants dominate, the experiment is mostly below its configured
pressure threshold. If deferrals grow but repeated grants stay zero, inspect
whether clears and new notifications progress rather than lowering thresholds
blindly. If migrations fall while throughput falls and PCIe RX rises, remote
access is likely becoming more expensive than the copies avoided.

Local validation uses production C code with ASan/UBSan and notification-service
integration harnesses. Full Linux module compilation and GPU throughput testing
must be performed on the target Linux machine.
