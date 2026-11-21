# Adaptive replacement admission

Build and reload the experiment branch with
`uvm_perf_adaptive_replacement_admission_enable=1` and bounded replacement
enabled. Keep indicated pages disabled to test whole-trigger-block admission.
Use the same module with the adaptive flag set to 0 as the control. For a
matched comparison of the adaptive rule against base admission, disable
coverage and extra eviction probation in both runs. Keep the access-counter
threshold, prefetch chunk, byte budget, and external reservation fixed.
Diagnostics can stay off for throughput measurements. Check sysfs after reload.

The experiment applies the base clear-separated observation count to
CPU-preferred candidates even below the occupancy cutoff. At the start of
probation it checks whether PMA free bytes plus reusable PMM roots can cover
one trigger root after the existing reserve and in-flight commitments. With
insufficient headroom it adds the current per-GPU penalty and one more
observation for a tracked return when the penalty is nonzero. The observation
target is frozen for that probation. The reservation remains authoritative.
The adaptive rule supersedes coverage probation while enabled.

The controller starts with no extra observations. For each group of 16
successful background evictions with valid VAB recency, increase the extra
target by one when at most four victims had any observed activity after
readiness, and decrease it by one when at least twelve did. The penalty ranges
from zero to two. Every sixteenth candidate that would receive a penalty uses
the base observation count as a probe. This is bounded exploration through
the existing freshness gate. The probe and ordinary candidates share the
same per-GPU feedback, so this version does not estimate a per-probe speedup.
The existing optional legacy eviction floor still applies if enabled.

VAB evidence is restricted to successfully processed background victims with
valid LRU recency. Use VAB mode 3 with verified mapping. A 4 MiB bit can reflect a
neighboring root, and the eviction sample is biased toward cold victims.
Consequently this signal measures observed activity among victims, not exact
reuse or migration benefit. It requires no per-notification root scan and
uses groups of observed events instead of a wall-clock control interval.

The monitor captures `adaptive_level` (gauge), `adaptive_feedback_samples`,
`adaptive_feedback_observed`, `adaptive_candidates`, and `adaptive_probes`.
The last two count elevated probation candidates before probe selection and
the subset selected as probes. Compare Sirius cold/warm query times and vLLM
warm throughput, plus population and allocated-root releases. Repeat paired
runs to distinguish small changes from run variation.
