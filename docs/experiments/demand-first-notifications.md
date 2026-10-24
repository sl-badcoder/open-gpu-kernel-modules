# Demand-first access-counter notifications

This experiment changes managed access-counter notification servicing so the
notified block is fully serviced and its flags are processed before speculative
neighbor prefetch begins. Speculative triggers are queued in a bounded,
stack-owned queue and drained synchronously after demand servicing for the same
VA-space/GPU segment. The queue also drains at capacity and before releasing
the VA-space read lock or changing GPU context.

The queue stores `uvm_va_block_t *` pointers. This is safe because
`service_notifications` retains the VA-space mmap lock and VA-space read lock
for the entire segment, and the queue is always drained or discarded before
either lock is released. No block reference is manufactured or needed.

Demand or notification-flag errors do not drain pending speculative entries;
they are discarded, so an error cannot cause speculative work after failed
demand servicing. Non-managed, HMM, ATS, missing-GPU, and missing-block paths
never enqueue entries. The existing prefetch helper remains responsible for
its tracker wait and eager-eviction admission/completion gate.

## Validation

On the macOS host, run:

```sh
python3 tests/experiments/test_demand_first_notifications.py
```

This brace-extracts the production queue helpers and `service_notifications`
orchestrator directly from `uvm_gpu_access_counters.c`, supplies mocked UVM
types and service functions, and compiles the resulting harness with
`-Werror`, AddressSanitizer, and UndefinedBehaviorSanitizer. It exercises
capacity flush, VA-space/GPU transitions, demand and flag errors, and null,
ATS, and non-managed notification groups. A Linux NVIDIA kernel build was not
available in this environment.
