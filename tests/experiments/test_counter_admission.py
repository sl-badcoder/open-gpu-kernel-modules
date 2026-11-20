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
_Static_assert(sizeof(uvm_counter_admission_region_t) == 16,
               "observation count must fit the existing region-state padding");

static void three_observations_require_two_distinct_completed_clears(void)
{
    uvm_counter_admission_region_t r = {0};
    assert(!uvm_counter_admission_decide(&r, true, 0, 10, 10, 3));
    assert(r.observations == 1 && r.clear_serial == 11);
    assert(!uvm_counter_admission_decide(&r, true, 0, 11, 10, 3));
    assert(!uvm_counter_admission_decide(&r, true, 0, 11, 11, 3));
    assert(r.observations == 2 && r.clear_serial == 12);
    for (int i = 0; i < 100; ++i)
        assert(!uvm_counter_admission_decide(&r, true, 0, 12, 11, 3));
    assert(r.observations == 2);
    assert(uvm_counter_admission_decide(&r, true, 0, 12, 12, 3));
    assert(!r.pending && !r.observations);
}

static void probation_restarts_and_large_clear_jumps_count_once(void)
{
    uvm_counter_admission_region_t r = {0};
    assert(!uvm_counter_admission_decide(&r, true, 0, 1, 1, 3));
    assert(!uvm_counter_admission_decide(&r, true, 0, 100, 100, 3));
    assert(r.observations == 2 && r.clear_serial == 101);
    assert(!uvm_counter_admission_decide(&r, true, 1, 101, 101, 3));
    assert(r.observations == 1 && r.buffer == 1);
    assert(!uvm_counter_admission_decide(&r, true, 1, 102, 102, 3));
    assert(uvm_counter_admission_decide(&r, false, 1, 102, 102, 3));
    assert(!r.pending && !r.observations);
    assert(!uvm_counter_admission_decide(&r, true, 1, 103, 103, 3));
    // Failed clear/eviction cancels pending in the production caller. Stale
    // observation bytes must never survive cancellation as usable evidence.
    r.pending = false; r.observations = 2;
    assert(!uvm_counter_admission_decide(&r, true, 1, 104, 104, 3));
    assert(r.observations == 1);
}

static void required_counts_are_bounded_and_do_not_advance_from_backlog(void)
{
    for (NvU32 config = 0; config <= 10; ++config) {
        NvU32 required = config < 2 ? 2 : (config > 8 ? 8 : config);
        uvm_counter_admission_region_t r = {0};
        for (NvU32 i = 0; i < required; ++i)
            assert(uvm_counter_admission_decide(&r, true, 0, 10 + i, 10 + i, config) ==
                   (i + 1 == required));
    }
    uvm_counter_admission_region_t r = {0};
    uvm_counter_admission_buffer_t b = {.submitted=10, .fresh=10, .batch_fresh=10};
    assert(!uvm_counter_admission_decide(&r, true, 0, b.submitted, b.batch_fresh, 3));
    ++b.submitted; // clear for observation 1
    uvm_counter_admission_prepare(&b, true, 2);
    assert(!uvm_counter_admission_decide(&r, true, 0, b.submitted, b.batch_fresh, 3));
    uvm_counter_admission_consumed(&b, 2);
    assert(!uvm_counter_admission_decide(&r, true, 0, b.submitted, b.batch_fresh, 3));
    assert(r.observations == 1); // crossing batch remains fenced
    uvm_counter_admission_prepare(&b, false, 0);
    assert(!uvm_counter_admission_decide(&r, true, 0, b.submitted, b.batch_fresh, 3));
    assert(r.observations == 2);
    ++b.submitted; // clear for observation 2
    uvm_counter_admission_prepare(&b, true, 0);
    assert(uvm_counter_admission_decide(&r, true, 0, b.submitted, b.batch_fresh, 3));
}

static void low_pressure_is_immediate_and_clears_pending(void)
{
    uvm_counter_admission_region_t r = {0};
    assert(uvm_counter_admission_decide(&r, true, 1, 10, 10, 2) == false);
    assert(r.pending && r.buffer == 1 && r.clear_serial == 11);
    assert(uvm_counter_admission_decide(&r, false, 1, 10, 10, 2) == true);
    assert(!r.pending);
    assert(uvm_counter_admission_decide(&r, true, 1, 11, 11, 2) == false);
}

static void pressure_requires_fresh_second_observation(void)
{
    uvm_counter_admission_region_t r = {0};
    assert(!uvm_counter_admission_decide(&r, true, 2, 20, 20, 2));
    /* Repeated notification from before the completed clear is insufficient. */
    assert(!uvm_counter_admission_decide(&r, true, 2, 20, 20, 2));
    assert(!uvm_counter_admission_decide(&r, true, 2, 21, 20, 2));
    /* A completed clear advances fresh to the recorded serial. */
    assert(uvm_counter_admission_decide(&r, true, 2, 21, 21, 2));
    assert(!r.pending);
}

static void different_buffers_do_not_satisfy_pending_region(void)
{
    uvm_counter_admission_region_t r = {0};
    assert(!uvm_counter_admission_decide(&r, true, 0, 4, 4, 2));
    assert(!uvm_counter_admission_decide(&r, true, 1, 5, 5, 2));
    assert(r.pending && r.buffer == 1 && r.clear_serial == 6);
    assert(!uvm_counter_admission_decide(&r, true, 0, 6, 6, 2));
    assert(!uvm_counter_admission_decide(&r, true, 1, 6, 5, 2));
    assert(uvm_counter_admission_decide(&r, true, 1, 7, 7, 2));
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
    const NvU64 edges[] = {64ULL<<20, 256ULL<<20, 1ULL<<30, 4ULL<<30};
    assert(uvm_counter_admission_distance_bucket(0)==0);
    for (unsigned i=0;i<4;++i) {
        assert(uvm_counter_admission_distance_bucket(edges[i]-1)==i);
        assert(uvm_counter_admission_distance_bucket(edges[i])==i+1);
    }
    assert(uvm_counter_admission_distance_bucket(UINT64_MAX)==4);
    uvm_counter_admission_history_t h = {.evicted=true, .population_at_eviction=100};
    assert(uvm_counter_admission_required(&h, 100, 64, 2, 2)==4);
    assert(uvm_counter_admission_required(&h, 163, 64, 2, 2)==4);
    assert(uvm_counter_admission_required(&h, 164, 64, 2, 2)==2);
    assert(uvm_counter_admission_required(&h, 100, 0, 2, 2)==2);
    assert(uvm_counter_admission_required(&h, 100, 64, 7, 999)==8);
    h.evicted=false;
    assert(uvm_counter_admission_required(&h, 100, 64, 2, 2)==2);
    uvm_counter_admission_region_t frozen={0};
    assert(!uvm_counter_admission_decide(&frozen,true,0,0,0,4));
    assert(!uvm_counter_admission_decide(&frozen,true,0,1,1,2));
    assert(!uvm_counter_admission_decide(&frozen,true,0,2,2,2));
    assert(uvm_counter_admission_decide(&frozen,true,0,3,3,2));
    three_observations_require_two_distinct_completed_clears();
    probation_restarts_and_large_clear_jumps_count_once();
    required_counts_are_bounded_and_do_not_advance_from_backlog();
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
    match = re.search(r"(?:static )?(?:bool|void) " + re.escape(name) + r"\(", text)
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


GATE_STUBS = r'''
#define clamp(a,b,c) ((a)<(b)?(b):((a)>(c)?(c):(a)))
typedef struct { uvm_counter_admission_region_t counter_admission; uvm_counter_admission_history_t counter_admission_history; unsigned resident; } uvm_va_block_gpu_state_t;
typedef struct { int dummy; } range_t;
typedef struct { int lock; range_t *managed_range; } uvm_va_block_t;
typedef struct { int id; long long counter_admission_population_bytes; } uvm_gpu_t;
static bool uvm_perf_access_counter_admission_eviction_enable=true;
static bool uvm_perf_access_counter_admission_distance_stats_enable=true;
static long long uvm_cpu_preferred_return_probations[5], uvm_cpu_preferred_return_grants[5];
static unsigned uvm_perf_access_counter_admission_eviction_mb=256, uvm_perf_access_counter_admission_eviction_extra=2;
static long long uvm_cpu_preferred_admission_eviction_probations, uvm_cpu_preferred_admission_eviction_deferred;
static long long atomic64_read(long long *p) {return *p;}
static bool enabled=true, pressure=true, have_state=true, fail_alloc=false;
static unsigned uvm_perf_access_counter_admission_observations=3;
static long long uvm_cpu_preferred_admission_requests, uvm_cpu_preferred_admission_deferred;
static long long uvm_cpu_preferred_admission_low_pressure, uvm_cpu_preferred_admission_repeated;
static long long uvm_cpu_preferred_admission_state_failures;
static uvm_va_block_gpu_state_t state;
static bool uvm_page_mask_empty(unsigned *p) {return !*p;}
static void atomic64_inc(long long *p) {++*p;}
static void uvm_assert_mutex_locked(int *p) {assert(*p);}
static bool uvm_cpu_block_policy_counter_admission_enabled(void) {return enabled;}
static bool uvm_va_block_is_hmm(uvm_va_block_t *b) {(void)b;return false;}
static bool uvm_cpu_block_policy_should_service_4k(range_t *r,uvm_gpu_t *g) {(void)r;(void)g;return true;}
static bool uvm_cpu_block_policy_counter_pressure(uvm_gpu_t *g) {(void)g;return pressure;}
static uvm_va_block_gpu_state_t *uvm_va_block_gpu_state_get(uvm_va_block_t *b,int id)
{(void)b;(void)id;return have_state?&state:0;}
static uvm_va_block_gpu_state_t *uvm_va_block_gpu_state_get_alloc(uvm_va_block_t *b,uvm_gpu_t *g)
{(void)b;(void)g;return fail_alloc?0:&state;}
'''

GATE_CASES = r'''
int main(void) {
    uvm_va_block_gpu_state_t lower={0}, upper={0};
    lower.counter_admission_history.evicted=true;
    lower.counter_admission_history.population_at_eviction=123;
    lower.counter_admission.pending=upper.counter_admission.pending=true;
    production_split_history(&lower,&upper);
    assert(upper.counter_admission_history.evicted);
    assert(upper.counter_admission_history.population_at_eviction==123);
    assert(!lower.counter_admission.pending && !upper.counter_admission.pending);
    range_t range={0};uvm_va_block_t block={.lock=1,.managed_range=&range};uvm_gpu_t gpu={0};
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,10,10));
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,11,11));
    assert(uvm_cpu_block_policy_admit_counter(&block,&gpu,0,12,12));
    assert(uvm_cpu_preferred_admission_requests==3&&uvm_cpu_preferred_admission_deferred==2);
    assert(uvm_cpu_preferred_admission_repeated==1);
    uvm_perf_access_counter_admission_observations=2;
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,13,13));
    assert(uvm_cpu_block_policy_admit_counter(&block,&gpu,0,14,14));
    uvm_perf_access_counter_admission_observations=3;
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,15,15));
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,16,16));
    pressure=false;
    assert(uvm_cpu_block_policy_admit_counter(&block,&gpu,0,16,16));
    assert(!state.counter_admission.pending&&!state.counter_admission.observations);
    assert(uvm_cpu_preferred_admission_low_pressure==1);
    pressure=true;
    uvm_perf_access_counter_admission_observations=2;
    state.counter_admission_history.evicted=true;
    state.counter_admission_history.population_at_eviction=0;
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,20,20));
    gpu.counter_admission_population_bytes=512LL*1024*1024;
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,21,21));
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,22,22));
    assert(uvm_cpu_block_policy_admit_counter(&block,&gpu,0,23,23));
    assert(uvm_cpu_preferred_admission_eviction_probations==1);
    assert(uvm_cpu_preferred_admission_eviction_deferred==3);
    gpu.counter_admission_population_bytes=0;
    uvm_perf_access_counter_admission_eviction_enable=false;
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,24,24));
    assert(uvm_cpu_block_policy_admit_counter(&block,&gpu,0,25,25));
    uvm_perf_access_counter_admission_eviction_enable=true;
    state.resident=1;state.counter_admission_history.evicted=true;
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,26,26));
    assert(!state.counter_admission_history.evicted && state.counter_admission.required==2);
    state.resident=0;
    // Instrumentation remains active with extra probation disabled.
    uvm_perf_access_counter_admission_eviction_enable=false;
    state.counter_admission.pending=false;
    state.counter_admission_history.evicted=true;
    state.counter_admission_history.population_at_eviction=0;
    gpu.counter_admission_population_bytes=1LL<<30;
    long long before_prob=uvm_cpu_preferred_return_probations[3];
    long long before_grant=uvm_cpu_preferred_return_grants[3];
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,30,30));
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,30,30));
    assert(uvm_cpu_preferred_return_probations[3]==before_prob+1);
    assert(uvm_cpu_block_policy_admit_counter(&block,&gpu,0,31,31));
    assert(uvm_cpu_preferred_return_grants[3]==before_grant+1);
    pressure=false;
    assert(uvm_cpu_block_policy_admit_counter(&block,&gpu,0,32,32));
    assert(uvm_cpu_preferred_return_grants[3]==before_grant+2);
    pressure=true;
    have_state=false;fail_alloc=true;
    assert(!uvm_cpu_block_policy_admit_counter(&block,&gpu,0,17,17));
    assert(uvm_cpu_preferred_admission_state_failures==1);
    assert(uvm_cpu_preferred_admission_requests==uvm_cpu_preferred_admission_deferred+
           uvm_cpu_preferred_admission_low_pressure+uvm_cpu_preferred_admission_repeated);
    puts("counter admission production gate: configurable observations, reset, accounting and state failure passed");
}
'''

def test_production_gate():
    source = (ROOT / "kernel-open/nvidia-uvm/uvm_cpu_block_policy.c").read_text()
    block_source = (ROOT / "kernel-open/nvidia-uvm/uvm_va_block.c").read_text()
    split = block_source.split("    new_gpu_state->counter_admission_history =", 1)[1].split(
        "    block_copy_split_gpu_chunks", 1)[0]
    split = "static void production_split_history(uvm_va_block_gpu_state_t *existing_gpu_state, uvm_va_block_gpu_state_t *new_gpu_state) {\n    new_gpu_state->counter_admission_history =" + split + "}\n"
    gpu_source = (ROOT / "kernel-open/nvidia-uvm/uvm_gpu.h").read_text()
    assert "counter_admission_population_bytes" in gpu_source.split("struct uvm_gpu_struct\n",1)[1].split("struct uvm_parent_gpu_struct\n",1)[0]
    code = PRELUDE + HEADER.read_text() + GATE_STUBS + split + extract_function(source, "counter_admission_record_return_grant") + extract_function(
        source, "uvm_cpu_block_policy_admit_counter") + GATE_CASES
    with tempfile.TemporaryDirectory(prefix="counter-admission-gate-") as directory:
        cfile, binary = Path(directory) / "test.c", Path(directory) / "test"
        cfile.write_text(code)
        subprocess.run(["cc", "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-Wno-unused-function", "-fsanitize=address,undefined",
                        str(cfile), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
    test_production_gate()
