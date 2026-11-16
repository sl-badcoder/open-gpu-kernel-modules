#!/usr/bin/env python3
"""Production-header tests for pressure-aware access-counter admission."""
from pathlib import Path
import subprocess
import tempfile
import re


ROOT = Path(__file__).parents[2]
HEADER = ROOT / "kernel-open/nvidia-uvm/uvm_counter_admission.h"

PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
typedef uint32_t NvU32;
typedef uint64_t NvU64;
/* Keep the Linux current identifier available to catch accidental collisions
 * with kernel names when this small header is compiled standalone. */
struct task_struct { int pid; };
static struct task_struct current_kernel_task __attribute__((unused));
#define current current_kernel_task
'''

CASES = r'''
static void low_pressure_is_immediate_and_clears_pending(void)
{
    uvm_counter_admission_region_t r = {0};
    assert(uvm_counter_admission_decide(&r, true, 1, 10, 10) == false);
    assert(r.pending && r.buffer == 1 && r.clear_serial == 11);
    assert(uvm_counter_admission_decide(&r, false, 1, 10, 10) == true);
    assert(!r.pending);
    assert(uvm_counter_admission_decide(&r, true, 1, 11, 11) == false);
}

static void pressure_requires_fresh_second_observation(void)
{
    uvm_counter_admission_region_t r = {0};
    assert(!uvm_counter_admission_decide(&r, true, 2, 20, 20));
    /* Repeated notification from before the completed clear is insufficient. */
    assert(!uvm_counter_admission_decide(&r, true, 2, 20, 20));
    assert(!uvm_counter_admission_decide(&r, true, 2, 21, 20));
    /* A completed clear advances fresh to the recorded serial. */
    assert(uvm_counter_admission_decide(&r, true, 2, 21, 21));
    assert(!r.pending);
}

static void different_buffers_do_not_satisfy_pending_region(void)
{
    uvm_counter_admission_region_t r = {0};
    assert(!uvm_counter_admission_decide(&r, true, 0, 4, 4));
    assert(!uvm_counter_admission_decide(&r, true, 1, 5, 5));
    assert(r.pending && r.buffer == 1 && r.clear_serial == 6);
    assert(!uvm_counter_admission_decide(&r, true, 0, 6, 6));
    assert(!uvm_counter_admission_decide(&r, true, 1, 6, 5));
    assert(uvm_counter_admission_decide(&r, true, 1, 7, 7));
}

static void prepare_without_queued_notifications_commits_immediately(void)
{
    uvm_counter_admission_buffer_t s = {.submitted = 9, .fresh = 3};
    uvm_counter_admission_prepare(&s, true, 0);
    assert(s.fence == 9 && s.fresh == 9 && !s.fencing && s.batch_fresh == 9);
    /* No new clear means the same serial is stable. */
    uvm_counter_admission_prepare(&s, true, 0);
    assert(s.fresh == 9 && s.batch_fresh == 9);
}

static void crossing_fence_is_conservative_until_queue_drains(void)
{
    uvm_counter_admission_buffer_t s = {.submitted = 100, .fresh = 90};
    uvm_counter_admission_prepare(&s, true, 3);
    assert(s.fence == 100 && s.remaining == 3 && s.fencing);
    /* The batch crossing the fence cannot use the new serial. */
    assert(s.batch_fresh == 90);
    uvm_counter_admission_consumed(&s, 2);
    assert(s.fresh == 90 && s.remaining == 1 && s.fencing);
    uvm_counter_admission_prepare(&s, true, 0);
    assert(s.batch_fresh == 90);
    uvm_counter_admission_consumed(&s, 1);
    assert(!s.fencing && s.remaining == 0 && s.fresh == 100);
    uvm_counter_admission_prepare(&s, false, 0);
    assert(s.batch_fresh == 100);
}

static void incomplete_clear_cannot_advance_and_overconsumption_is_safe(void)
{
    uvm_counter_admission_buffer_t s = {.submitted = 7, .fresh = 7};
    uvm_counter_admission_prepare(&s, false, 4);
    assert(s.fresh == 7 && !s.fencing && s.batch_fresh == 7);
    s.submitted = 8;
    uvm_counter_admission_prepare(&s, true, 4);
    assert(s.fence == 8 && s.remaining == 4 && s.fencing);
    uvm_counter_admission_consumed(&s, 99);
    assert(!s.fencing && s.remaining == 0 && s.fresh == 8);
}

static void queued_count_boundary_and_wrap_abstraction(void)
{
    uvm_counter_admission_buffer_t s = {.submitted = 0xfffffff0ULL, .fresh = 0xffffffe0ULL};
    /* The caller supplies the ring distance, including a wrap-around result. */
    uvm_counter_admission_prepare(&s, true, 0xffffffffU);
    assert(s.remaining == 0xffffffffU && s.fence == 0xfffffff0ULL);
    uvm_counter_admission_consumed(&s, 0xfffffffeU);
    assert(s.remaining == 1 && s.fencing && s.fresh == 0xffffffe0ULL);
    uvm_counter_admission_consumed(&s, 1);
    assert(!s.fencing && s.fresh == 0xfffffff0ULL);
}

int main(void)
{
    low_pressure_is_immediate_and_clears_pending();
    pressure_requires_fresh_second_observation();
    different_buffers_do_not_satisfy_pending_region();
    prepare_without_queued_notifications_commits_immediately();
    crossing_fence_is_conservative_until_queue_drains();
    incomplete_clear_cannot_advance_and_overconsumption_is_safe();
    queued_count_boundary_and_wrap_abstraction();
    puts("counter admission: pressure gating, clear fences, buffer isolation and wrap-distance handling passed");
    return 0;
}
'''

FETCH_PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
typedef uint32_t NvU32;
typedef uint64_t NvU64;
struct task_struct { int pid; };
static struct task_struct current_kernel_task __attribute__((unused));
#define current current_kernel_task
typedef struct { int held; } lock_t;
typedef struct { int complete; } tracker_t;
typedef struct { lock_t clear_tracker_lock; tracker_t clear_tracker; } counters_t;
typedef struct uvm_parent_gpu_struct { counters_t access_counters; } uvm_parent_gpu_t;
typedef struct { NvU32 *pAccessCntrBufferPut; } rm_info_t;
typedef struct {
    uvm_parent_gpu_t *parent_gpu;
    rm_info_t rm_info;
    NvU32 max_notifications;
    NvU32 cached_get;
    NvU32 cached_put;
    uvm_counter_admission_buffer_t admission;
} uvm_access_counter_buffer_t;
static bool enabled;
static unsigned tracker_calls, mmio_reads, lock_calls;
static bool uvm_cpu_block_policy_counter_admission_enabled(void) { return enabled; }
static void uvm_mutex_lock(lock_t *lock) { assert(!lock->held); lock->held = 1; ++lock_calls; }
static void uvm_mutex_unlock(lock_t *lock) { assert(lock->held); lock->held = 0; ++lock_calls; }
static bool uvm_tracker_is_completed(tracker_t *tracker) { ++tracker_calls; return tracker->complete; }
static NvU32 read_put(NvU32 *put) { ++mmio_reads; return *put; }
#define UVM_GPU_READ_ONCE(x) read_put(&(x))
'''

FETCH_TYPES = r'''
#include <stdbool.h>
#include <stdint.h>
typedef uint32_t NvU32;
typedef uint64_t NvU64;
'''


def extract_function(text: str, name: str) -> str:
    match = re.search(r"static bool " + re.escape(name) + r"\(", text)
    if not match:
        raise ValueError(name)
    brace = text.index("{", match.start())
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start():pos + 1]
    raise ValueError(name)


FETCH_CASES = r'''
static void fetch_tests(void)
{
    NvU32 put = 2;
    uvm_parent_gpu_t parent = {0};
    uvm_access_counter_buffer_t a = {
        .parent_gpu = &parent, .rm_info = {.pAccessCntrBufferPut = &put},
        .max_notifications = 8, .cached_get = 6,
        .admission = {.submitted = 9, .fresh = 8, .tracking = true},
    };

    enabled = false; tracker_calls = mmio_reads = lock_calls = 0;
    assert(!counter_admission_prepare_fetch(&a));
    assert(!tracker_calls && !mmio_reads && !lock_calls);

    enabled = true; a.admission.tracking = false;
    assert(!counter_admission_prepare_fetch(&a));
    assert(!tracker_calls && !mmio_reads);
    a.admission.tracking = true;

    parent.access_counters.clear_tracker.complete = 0;
    assert(!counter_admission_prepare_fetch(&a));
    assert(tracker_calls == 1 && mmio_reads == 0 && a.admission.fresh == 8);

    parent.access_counters.clear_tracker.complete = 1;
    assert(counter_admission_prepare_fetch(&a));
    assert(tracker_calls == 2 && mmio_reads == 1 && lock_calls == 4);
    assert(a.cached_put == 2 && a.admission.fence == 9);
    assert(a.admission.remaining == 4 && a.admission.fencing);

    /* An active fence avoids another tracker query and MMIO PUT read. */
    a.admission.submitted = 12;
    put = 7;
    assert(!counter_admission_prepare_fetch(&a));
    assert(tracker_calls == 2 && mmio_reads == 1 && a.cached_put == 2);

    /* A completed fence can be consumed and the next clear reads PUT once. */
    uvm_counter_admission_consumed(&a.admission, 4);
    a.admission.submitted = 13;
    a.admission.fresh = 9;
    assert(counter_admission_prepare_fetch(&a));
    assert(tracker_calls == 3 && mmio_reads == 2 && a.cached_put == 7);
    assert(a.admission.remaining == 1);
}

int main(void)
{
    fetch_tests();
    puts("counter admission fetch: tracker ordering, cached PUT, fencing and ring distance passed");
    return 0;
}
'''


def main() -> None:
    source = PRELUDE + HEADER.read_text() + CASES
    with tempfile.TemporaryDirectory(prefix="counter-admission-") as directory:
        root = Path(directory)
        cfile = root / "test.c"
        binary = root / "test"
        cfile.write_text(source)
        subprocess.run(
            ["cc", "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
             "-fsanitize=address,undefined", str(cfile), "-o", str(binary)],
            check=True,
        )
        subprocess.run([str(binary)], check=True)

    access_source = (ROOT / "kernel-open/nvidia-uvm/uvm_gpu_access_counters.c").read_text()
    fetch_source = FETCH_TYPES + HEADER.read_text() + FETCH_PRELUDE + extract_function(
        access_source, "counter_admission_prepare_fetch") + FETCH_CASES
    with tempfile.TemporaryDirectory(prefix="counter-admission-fetch-") as directory:
        root = Path(directory)
        cfile = root / "test.c"
        binary = root / "test"
        cfile.write_text(fetch_source)
        subprocess.run(
            ["cc", "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
             "-fsanitize=address,undefined", str(cfile), "-o", str(binary)],
            check=True,
        )
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
