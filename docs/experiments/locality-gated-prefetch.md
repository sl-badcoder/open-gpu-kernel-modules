# Locality-gated access-counter prefetch

Base: `595.71.05/experiment/demand-paced-movement` at `e91238ed`.

This experiment keeps demand migration independent from speculative movement.
Access-counter notifications still migrate the pages in the original signal.
Neighbor prefetch is admitted only when the signal covers at least 25 percent
of its VA block. The threshold is configurable with
`uvm_perf_prefetch_min_signal_density`. Sparse signals therefore do not spend
interconnect bandwidth on neighboring 2 MiB blocks.

The gate does not reduce the cost of full-block promotion within the trigger
block when promotion is enabled. The trigger block remains subject to the
existing promotion policy; this experiment only controls neighboring blocks.

The classifier runs before optional full-block promotion expands the mask.
Density is only a conservative locality signal; it does not prove sequential
execution. Notification order is address-sorted, so this branch does not infer
stream direction from notification order. Directional prediction and a stream
frontier are intentionally left for a later experiment.

Check the loaded value and movement counters with:

```sh
cat /sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_min_signal_density
cat /sys/module/nvidia_uvm/parameters/uvm_cpu_preferred_prefetched_bytes
cat /sys/module/nvidia_uvm/parameters/uvm_cpu_preferred_range_prefetches
```

Compare sparse random, dense sequential and mixed workloads. Record demand and
speculative transfer bytes, D2H eviction bytes, refaults and runtime. This is
an experiment, not a claim that a fixed density threshold is optimal.
