# Independent UVM performance experiments

All four branches fork directly from `f735561b` (the tracked source is identical
to the faster `2c5225fd`). They are siblings, not a stack. The rejected
`2f4de8ef` optimization is absent. Test each against this baseline before
combining winners.

| Branch under `595.71.05/experiment/` | Isolated change | Regression test |
| --- | --- | --- |
| `batched-pma-free` | One PMA array free per completed eviction submission batch | `test_batched_pma_free.py` |
| `demand-first-notifications` | Service up to eight notified blocks before speculative prefetch; drain on GPU/VA-space boundaries | `test_demand_first_notifications.py` |
| `batched-eviction-selection` | Bounded snapshots of used eviction candidates, with eligibility revalidation | `test_batched_eviction_selection.py` |
| `coalesced-prefetch` | Skip overlap with concurrently active windows in the same managed range/GPU | `test_coalesced_prefetch.py` |

The original `prefetch-after-low-watermark` branch remains the baseline.
Coalescing is in-flight only; it does not cache completed residency. Skipped
triggers remain notification-retryable if an older owner stops early. It can
add overhead when all notifications are processed serially. Candidate ranking
is sampled within a bounded window, so concurrent heat changes can affect
which roots are chosen compared with the baseline.

## Build and compare

Fetch the branches and check out one at a time. For example:

```sh
git fetch origin
git switch --track origin/595.71.05/experiment/batched-pma-free
python3 tests/experiments/test_batched_pma_free.py
make modules -j"$(nproc)" DEBUG=0
```

If the local branch already exists, use `git switch` with its local name.
Install/load each build using your normal driver procedure and the repository
README. Building or fetching alone does not change the running module. Use
matching 595.71.05 user-space/GSP components. Record the source commit,
`modinfo -F srcversion nvidia_uvm`, kernel version, GPU/VRAM, loaded module
parameters, and workload command for each result. Use the same build mode and
workload for every branch. `DEBUG=1` enables extra checks/logging and should not
be mixed with release performance results.

Use one unmeasured warm-up and at least five timed repetitions, comparing the
median and spread. Alternate baseline and experiment runs to expose thermal or
clock drift. Keep cold-residency and warm-residency tests separate; rerunning a
command does not necessarily recreate the same residency. Synchronize GPU work
before the workload timer stops. Keep unrelated GPU work idle.

The tests compile extracted production C with mocked external kernel/GPU
operations and ASan/UBSan. They are not Linux kernel builds or GPU performance
measurements. Linux module build, load, mapping/coherence and runtime tests are
still required. Run the branch-specific test directly with Python; pytest is
not required.

## First pass: preserve the existing numbers

For the initial branch comparison, keep your current known-good loaded
parameters. The source defaults for this policy are:

| Module parameter | Value | Role |
| --- | ---: | --- |
| `uvm_cpu_preferred_blocks_enable` | 0 | Custom CPU-preferred policy off by default; keep **1** if that is your current experiment setup |
| `uvm_cpu_preferred_block_promotion_enable` | 1 | Promote the notified populated VA block |
| `uvm_cpu_preferred_prefetch_chunk_mb` | 64 | Speculative window in MiB, rounded to 2 MiB blocks |
| `uvm_perf_eager_eviction_enable` | 1 | Eager eviction enabled |
| `uvm_perf_eager_eviction_high_watermark` | 90 | Start forced recovery at this reported usage percent |
| `uvm_perf_eager_eviction_low_watermark` | 85 | Resume prefetch after recovery reaches this level |
| `uvm_perf_eager_eviction_batch_count` | 16 | Maximum roots submitted per worker invocation |
| `uvm_perf_eager_eviction_interval_ms` | 25 | Aging interval and recovery backoff |
| `uvm_perf_eager_eviction_heat_ratio` | 2 | Heat contrast for non-forced replacement |

One root is 2 MiB. `batch_count=16` means up to 32 MiB per pass, **not** a
single 32 MiB DMA transfer. Productive recovery is immediately rescheduled.
The implementation limits outstanding submissions to 16 roots even when
`batch_count` is increased: 32 or 64 allows additional bounded submission/drain
rounds in one invocation. The PMA-free and candidate-snapshot experiments also
use bounded arrays; their limits are not independently tunable parameters.
The demand-first queue and coalescing windows are fixed at eight entries.

## Second pass: tune one variable at a time

There is no hardware-independent best configuration. These are hypotheses to
measure, not guaranteed faster presets. NVIDIA's oversubscription study also
shows that access pattern, reuse and platform substantially change outcomes:
https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/

1. **Prefetch window first.** For sequential access with reuse, sweep
   `32, 64, 128` MiB, starting at your existing 64. My first additional candidate
   is **32 MiB** if unnecessary migration/eviction traffic is high. For sparse
   or random access, try `0, 8, 16` MiB. Zero disables speculative chunk prefetch;
   it does not disable full-block promotion when promotion remains enabled.
2. **Eviction batch next.** Sweep `8, 16, 32, 64`, starting at 16. Try **32**
   after identifying the best branch/window if recovery scheduling overhead
   remains substantial. Larger values reduce worker scheduling boundaries but
   can increase service latency; they do not remove underlying copy cost.
3. **Watermarks afterward.** Start at **90/85**. Compare `90/88` if long pauses
   dominate and there is enough measured headroom, or `88/82` if allocations
   repeatedly run out of space. Narrow gaps increase phase-switch frequency;
   wider gaps reclaim more VRAM and keep prefetch paused longer. Keep low below
   high, and account for other VRAM consumers.
4. **Leave aging/grace alone initially.** Keep interval **25 ms**, heat ratio
   **2**, and the migration-aware grace parameters unchanged. Changing the
   interval also changes heat-aging and stale-interval timing; it is not simply
   an eviction throughput knob. If later needed, compare `10` and `25` ms in a
   separate experiment after the other variables are fixed.

The chunk size and promotion toggle are writable at runtime in this source.
The watermarks, batch count, aging interval and heat ratio are load-time
parameters, requiring a genuine module reload. `modprobe` on an already-loaded
module will not update them. For a freshly loaded custom-policy experiment,
a conservative starting parameter set is:

```text
uvm_cpu_preferred_blocks_enable=1
uvm_cpu_preferred_block_promotion_enable=1
uvm_cpu_preferred_prefetch_chunk_mb=64
uvm_perf_eager_eviction_enable=1
uvm_perf_eager_eviction_high_watermark=90
uvm_perf_eager_eviction_low_watermark=85
uvm_perf_eager_eviction_batch_count=16
uvm_perf_eager_eviction_interval_ms=25
uvm_perf_eager_eviction_heat_ratio=2
```

Report runtime together with H2D/D2H transfer volume, remote-access activity,
and fault/notification latency when your profiler supports them. Faster
migration alone is not proof of faster application execution. The baseline
exports promotion/prefetch counts and prefetched bytes, but not the extra
reclamation/demotion counters referenced by some external helper scripts.
