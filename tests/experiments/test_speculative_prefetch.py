#!/usr/bin/env python3
"""Execute the production prefetch walker and churn hooks with mocked UVM I/O."""
from pathlib import Path
import subprocess
import tempfile
from test_demand_first_eviction import function

ROOT = Path(__file__).parents[2]

PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
typedef unsigned long long NvU64; typedef unsigned short NvU16;
typedef int NV_STATUS;
typedef struct { int marker; } uvm_prefetch_refault_block_t;
#define NV_OK 0
#define UVM_SIZE_1MB (1ULL << 20)
#define UVM_VA_BLOCK_SIZE (2ULL << 20)
#define NSEC_PER_MSEC 1000000ULL
#define UVM_ASSERT(x) assert(x)
#define UVM_ALIGN_DOWN(x,a) ((x) & ~((NvU64)(a)-1))
#define ARRAY_SIZE(x) (sizeof(x)/sizeof((x)[0]))
#define READ_ONCE(x) (x)
#define min(a,b) ((a)<(b)?(a):(b))
#define min_t(type,a,b) ((a)<(b)?(a):(b))
#define UVM_DBG_PRINT(...) do {} while(0)
struct mm_struct { int unused; };
typedef struct { int lock; } uvm_va_space_t;
typedef struct { bool pending; int buffer; NvU64 clear_serial; } counter_admission_t;
typedef struct { NvU64 speculative_prefetch_ns, speculative_evicted_ns, speculative_cohort; uvm_prefetch_refault_block_t refault; counter_admission_t counter_admission; struct { NvU64 population_at_eviction; bool evicted; } counter_admission_history; unsigned resident; } uvm_va_block_gpu_state_t;
typedef struct { int unused; } uvm_pmm_gpu_t;
typedef struct { NvU64 bytes; NvU64 generation; } uvm_prefetch_reservation_t;
typedef struct { int id; uvm_pmm_gpu_t pmm; long long counter_admission_population_bytes; } uvm_gpu_t;
typedef struct uvm_va_range_managed_struct uvm_va_range_managed_t;
typedef struct {
 uvm_va_range_managed_t *managed_range; NvU64 start; int lock;
 unsigned gpu_prefetch_started, gpu_prefetch_retry; uvm_va_block_gpu_state_t state;
} uvm_va_block_t;
struct uvm_va_range_managed_struct {
 struct { uvm_va_space_t *va_space; } va_range;
 NvU64 start; NvU64 cpu_prefetch_range_id; bool cpu_access_counter_policy; size_t count; uvm_va_block_t *blocks[64];
};
typedef struct { int unused; } uvm_va_block_context_t;
typedef struct { int unused; } uvm_tracker_t;
#define UVM_TRACKER_INIT() {0}
static NvU64 now = 1000000000ULL;
static bool uvm_perf_access_counter_admission_eviction_enable=true;
static bool uvm_perf_access_counter_admission_distance_stats_enable=false;
static long long atomic64_read(long long *p) {return *p;}
static bool uvm_page_mask_empty(unsigned *p) {return !*p;}
static unsigned uvm_cpu_preferred_prefetch_chunk_mb;
static long long uvm_cpu_preferred_prefetch_capacity_stops, uvm_cpu_preferred_range_prefetches;
static long long uvm_cpu_preferred_prefetch_retries, uvm_cpu_preferred_prefetched_bytes;
static long long uvm_cpu_preferred_trigger_prefetched_bytes, uvm_cpu_preferred_neighbor_prefetched_bytes;
static void atomic64_inc(long long *p) {++*p;}
static void atomic64_add(NvU64 n,long long *p) {*p+=(long long)n;}
static bool enabled, supported, alloc_fail, trigger_resident;
static bool pressure;
static bool refault_enabled;
static unsigned refault_neighbor_count;
static NvU64 grant, requested, reserved, transferred;
static bool completed_success;
static NvU64 reservation_generation = 17;
static int calls, begin_calls, end_calls, waits, capacity_at, error_at, wait_error;
static int context_allocations, context_frees;
static int pressure_after_copies;
static int watermark_at, watermark_calls, feedback_quick, feedback_capacity;
static int pressure_pauses;
static size_t visited[64];
static size_t trigger_slot;
static uvm_va_space_t space;
static uvm_va_range_managed_t range;
static uvm_va_block_t blocks[64];
static uvm_gpu_t gpu;
static struct mm_struct mm;
static uvm_va_block_context_t context;
static NvU64 NV_GETTIME(void) {return now;}
static bool uvm_pmm_gpu_speculative_prefetch_enabled(void) {return enabled;}
static bool should_handle_gpu_signal(uvm_va_block_t *b,uvm_gpu_t *g) {(void)b;(void)g;return supported;}
static void uvm_assert_rwsem_locked_read(int *p) {assert(*p);}
static void uvm_mutex_lock(int *p) {assert(!*p);*p=1;}
static void uvm_mutex_unlock(int *p) {assert(*p);*p=0;}
static void uvm_assert_mutex_locked(int *p) {assert(*p);}
static uvm_va_block_gpu_state_t *uvm_va_block_gpu_state_get(uvm_va_block_t *b,int id) {(void)id;assert(b->lock);return &b->state;}
static bool uvm_cpu_block_policy_counter_pressure(uvm_gpu_t *g) {(void)g; return pressure || (pressure_after_copies>0 && calls>=pressure_after_copies);}
static NvU64 uvm_pmm_gpu_prefetch_feedback_window_ns(void) {return 20*NSEC_PER_MSEC;}
static void uvm_pmm_gpu_prefetch_pressure_pause(uvm_pmm_gpu_t *p) {(void)p;++pressure_pauses;}
static void uvm_pmm_gpu_prefetch_return(uvm_pmm_gpu_t *p,NvU64 cohort,NvU64 t) {(void)p;assert(t==now);if (cohort) {assert(cohort==42);++feedback_quick;}}
static void uvm_pmm_gpu_prefetch_allocation_failure(uvm_pmm_gpu_t *p) {(void)p;++feedback_capacity;}
static bool uvm_pmm_gpu_refault_enabled(void) { return refault_enabled; }
static unsigned uvm_pmm_gpu_refault_neighbors(uvm_pmm_gpu_t *p, NvU64 range_id, NvU64 address) { (void)p; assert(range_id == range.cpu_prefetch_range_id); assert(address == blocks[trigger_slot].start); return refault_neighbor_count; }
static void uvm_pmm_gpu_refault_evict(uvm_pmm_gpu_t *p, uvm_prefetch_refault_block_t *state) { (void)p; assert(state); }
static void __attribute__((unused)) uvm_pmm_gpu_refault_population(uvm_pmm_gpu_t *p, NvU64 range_id, NvU64 address) { (void)p; assert(range_id == range.cpu_prefetch_range_id); assert(address < range.count * UVM_VA_BLOCK_SIZE); }
static bool uvm_processor_mask_test(unsigned *p,int id) {(void)id;return *p;}
static bool uvm_processor_mask_test_and_clear_atomic(unsigned *p,int id) {(void)id;bool old=*p;*p=0;return old;}
static bool uvm_processor_mask_test_and_set_atomic(unsigned *p,int id) {(void)id;bool old=*p;*p=1;return old;}
static void uvm_processor_mask_clear_atomic(unsigned *p,int id) {(void)id;*p=0;}
static void uvm_processor_mask_set_atomic(unsigned *p,int id) {(void)id;*p=1;}
static uvm_va_block_context_t *uvm_va_block_context_alloc(struct mm_struct *m) {(void)m;++context_allocations;assert(reserved && begin_calls>0);return alloc_fail?NULL:&context;}
static void uvm_va_block_context_free(uvm_va_block_context_t *p) {assert(p==&context);assert(!reserved);++context_frees;}
static size_t uvm_va_range_num_blocks(uvm_va_range_managed_t *r) {return r->count;}
static size_t uvm_va_range_block_index(uvm_va_range_managed_t *r,NvU64 start) {return start/UVM_VA_BLOCK_SIZE-r->start/UVM_VA_BLOCK_SIZE;}
static uvm_va_block_t *uvm_va_range_block(uvm_va_range_managed_t *r,size_t index) {assert(index<r->count);return r->blocks[index];}
static bool uvm_pmm_gpu_eager_eviction_watermark_reached(uvm_pmm_gpu_t *p) {(void)p;return watermark_at>=0 && watermark_calls++>=watermark_at;}
static NvU64 uvm_pmm_gpu_eager_eviction_prefetch_begin(uvm_pmm_gpu_t *p,NvU64 bytes,uvm_prefetch_reservation_t *r) {
 (void)p;++begin_calls;assert(!reserved);requested=bytes;reserved=grant;assert(grant<=bytes);
 r->bytes=grant;r->generation=reservation_generation;return grant;
}
static void uvm_pmm_gpu_eager_eviction_prefetch_end(uvm_pmm_gpu_t *p,uvm_prefetch_reservation_t *r,NvU64 moved,bool success) {
 (void)p;assert(waits==1 && reserved==r->bytes && moved<=r->bytes);
 // OOM must retain the trigger claim until its reservation is released.
 if (alloc_fail) assert(blocks[trigger_slot].gpu_prefetch_started);
 assert(r->generation==reservation_generation);completed_success=success;reserved=0;transferred=moved;++end_calls;
}
static NV_STATUS uvm_tracker_wait_deinit(uvm_tracker_t *p) {(void)p;++waits;return wait_error;}
static NV_STATUS prefetch_block(uvm_va_block_t *b,uvm_gpu_t *g,struct mm_struct *m,
 uvm_va_block_context_t *ctx,uvm_tracker_t *tracker,NvU64 *moved,uvm_prefetch_reservation_t *reservation,bool *capacity,bool neighbor) {
 assert(reservation->generation==reservation_generation);
 (void)g;(void)m;(void)ctx;(void)tracker;assert(!b->lock && reserved && calls<64);
 visited[calls]=(size_t)(b-blocks);assert(neighbor == (b!=&blocks[trigger_slot]));
 ++calls;*capacity=(calls==capacity_at);
 if(*capacity)return 7;
 if(trigger_resident && !neighbor)return NV_OK;
 *moved+=UVM_VA_BLOCK_SIZE; assert(*moved<=reserved);
 return calls==error_at ? 8 : NV_OK;
}
'''

CASES = r'''
static void reset(void) {
 now=1000000000ULL;completed_success=false;
 memset(blocks,0,sizeof(blocks));memset(&range,0,sizeof(range));
 space.lock=1;range.va_range.va_space=&space;range.count=8;
 for(size_t i=0;i<8;++i){blocks[i].managed_range=&range;blocks[i].start=i*UVM_VA_BLOCK_SIZE;range.blocks[i]=&blocks[i];}
 enabled=supported=true;alloc_fail=trigger_resident=false;grant=4*UVM_SIZE_1MB;requested=reserved=transferred=0;
 refault_enabled=false;refault_neighbor_count=0;range.cpu_prefetch_range_id=77;range.cpu_access_counter_policy=true;
 calls=begin_calls=end_calls=waits=capacity_at=error_at=wait_error=0;context_allocations=context_frees=0;
 watermark_at=-1;watermark_calls=feedback_quick=feedback_capacity=pressure_pauses=0;
 trigger_slot=4;
 uvm_cpu_preferred_prefetch_chunk_mb=64;
 uvm_cpu_preferred_prefetch_capacity_stops=uvm_cpu_preferred_range_prefetches=0; pressure=false; pressure_after_copies=0;
 uvm_cpu_preferred_prefetch_retries=uvm_cpu_preferred_prefetched_bytes=0;
 uvm_cpu_preferred_trigger_prefetched_bytes=uvm_cpu_preferred_neighbor_prefetched_bytes=0;
}
static void run(void){uvm_cpu_block_policy_prefetch_on_signal(&blocks[trigger_slot],&gpu,&mm,0);
 assert(uvm_cpu_preferred_prefetched_bytes==uvm_cpu_preferred_trigger_prefetched_bytes+uvm_cpu_preferred_neighbor_prefetched_bytes);}
int main(void) {
 // Actual production loop: range clamps request; grant truncates it; trigger is first.
 reset();run();assert(requested==16*UVM_SIZE_1MB && calls==2 && visited[0]==4 && visited[1]==3);
 assert(uvm_cpu_preferred_trigger_prefetched_bytes==UVM_VA_BLOCK_SIZE && uvm_cpu_preferred_neighbor_prefetched_bytes==UVM_VA_BLOCK_SIZE);
 // Counter pressure admits only the triggering block, even when a learned
 // region mask requests additional neighbors.
 reset(); pressure=true; grant=2*UVM_SIZE_1MB; run();
 assert(calls==1 && visited[0]==4 && transferred==UVM_VA_BLOCK_SIZE);
 assert(end_calls==1 && completed_success && transferred==grant && !reserved);
 assert(uvm_cpu_preferred_trigger_prefetched_bytes==UVM_VA_BLOCK_SIZE && !uvm_cpu_preferred_neighbor_prefetched_bytes);
 reset();pressure=true;grant=2*UVM_SIZE_1MB;
 uvm_cpu_block_policy_prefetch_on_signal(&blocks[4],&gpu,&mm,0x12);
 assert(calls==1 && visited[0]==4 && requested==UVM_VA_BLOCK_SIZE);
 reset();grant=8*UVM_SIZE_1MB;pressure_after_copies=1;run();
 assert(calls==1 && visited[0]==4 && end_calls==1 && completed_success);
 assert(blocks[4].gpu_prefetch_retry && !reserved);
 reset();grant=8*UVM_SIZE_1MB;run();assert(calls==4 && visited[0]==4 && visited[1]==2 && visited[2]==3 && visited[3]==5);
 reset();grant=8*UVM_SIZE_1MB;trigger_slot=7;run();assert(calls==4 && visited[0]==7 && visited[1]==4 && visited[2]==5 && visited[3]==6);
 // A resident triggering block does not prevent neighboring transfers.
 reset();trigger_resident=true;run();assert(calls==2 && transferred==UVM_VA_BLOCK_SIZE);
 assert(!uvm_cpu_preferred_trigger_prefetched_bytes && uvm_cpu_preferred_neighbor_prefetched_bytes==UVM_VA_BLOCK_SIZE);
 // Sparse neighbor blocks consume no bytes and do not instantiate new blocks.
 reset();range.blocks[3]=NULL;run();assert(calls==1 && transferred==UVM_VA_BLOCK_SIZE && end_calls==1);
 // Transient denial retries, permanent disable never admits or sets retry.
 reset();grant=0;run();assert(!calls && !end_calls && blocks[4].gpu_prefetch_retry && !context_allocations && !context_frees);
 for(int i=0;i<100;++i)run();
 assert(!context_allocations && !context_frees && !reserved);
 reset();enabled=false;run();assert(!begin_calls && !calls && !waits && !blocks[4].gpu_prefetch_retry);
 reset();uvm_cpu_preferred_prefetch_chunk_mb=0;run();assert(!begin_calls && !calls);
 // Every admitted failure waits before releasing the reservation.
 reset();error_at=1;run();assert(calls==1 && end_calls==1 && !completed_success && !reserved);
 reset();capacity_at=2;run();assert(calls==2 && end_calls==1 && feedback_capacity==1 && !completed_success && !pressure_pauses && blocks[4].gpu_prefetch_retry);
 reset();watermark_at=0;run();assert(!calls && end_calls==1 && completed_success && !feedback_capacity && pressure_pauses==1);
 reset();wait_error=9;run();assert(end_calls==1 && !completed_success && !reserved);
 // A granted reservation is released on context OOM, and a later signal can retry.
 reset();alloc_fail=true;run();
 assert(begin_calls==1 && end_calls==1 && !completed_success && !reserved);
 assert(context_allocations==1 && !context_frees && !calls && !blocks[4].gpu_prefetch_started);
 alloc_fail=false;waits=0;run();
 assert(begin_calls==2 && end_calls==2 && completed_success && !reserved);
 assert(context_allocations==2 && context_frees==1 && calls==2);
 // Already claimed blocks and empty plans allocate no context.
 reset();blocks[4].gpu_prefetch_started=1;run();assert(!context_allocations && !begin_calls);
 // One-shot callback avoids repeat copies; capacity-retry can be claimed once.
 reset();run();run();assert(begin_calls==1 && calls==2 && end_calls==1 && !reserved);
 reset();grant=0;run();grant=4*UVM_SIZE_1MB;waits=0;run();assert(end_calls==1 && calls==2 && uvm_cpu_preferred_prefetch_retries==1);
 // Partial eviction does not create history. Complete eviction stamps once.
 reset();blocks[4].lock=1;blocks[4].state.resident=1;gpu.counter_admission_population_bytes=100;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu,true);
 assert(!blocks[4].state.counter_admission_history.evicted);
 blocks[4].state.resident=0;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu,true);
 assert(blocks[4].state.counter_admission_history.evicted);
 assert(blocks[4].state.counter_admission_history.population_at_eviction==100);
 gpu.counter_admission_population_bytes=200;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu,true);
 assert(blocks[4].state.counter_admission_history.population_at_eviction==200);
 blocks[4].state.counter_admission_history.evicted=false;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu,false);
 assert(!blocks[4].state.counter_admission_history.evicted);
 blocks[4].lock=0;
 reset();blocks[4].lock=1;uvm_perf_access_counter_admission_eviction_enable=false;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu,true);
 assert(!blocks[4].state.counter_admission_history.evicted);
 blocks[4].lock=0;uvm_perf_access_counter_admission_eviction_enable=true;
 // Actual eviction hook and actual demand callback: quick return is consumed once.
 reset();blocks[4].state.speculative_cohort=42;blocks[4].state.speculative_prefetch_ns=now-10;blocks[4].lock=1;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu,true);blocks[4].lock=0;
 assert(blocks[4].state.speculative_evicted_ns==now && !blocks[4].state.speculative_prefetch_ns);
 grant=0;run();run();assert(feedback_quick==1 && !blocks[4].state.speculative_evicted_ns && !blocks[4].state.speculative_cohort);
 // Expired returns and expired prefetch-to-eviction intervals cannot qualify.
 reset();blocks[4].state.speculative_evicted_ns=now-21*NSEC_PER_MSEC;grant=0;run();assert(!feedback_quick);
 reset();blocks[4].state.speculative_prefetch_ns=now-21*NSEC_PER_MSEC;blocks[4].lock=1;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu,true);blocks[4].lock=0;grant=0;run();assert(!feedback_quick);
 // Boundary is inclusive for each interval; a second partial eviction cannot refresh it.
 reset();blocks[4].state.speculative_cohort=42;
 blocks[4].state.speculative_prefetch_ns=now-20*NSEC_PER_MSEC;blocks[4].lock=1;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu,true);blocks[4].lock=0;
 NvU64 first_eviction=blocks[4].state.speculative_evicted_ns;
 assert(first_eviction==now);
 now+=10*NSEC_PER_MSEC;blocks[4].lock=1;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu,true);blocks[4].lock=0;
 assert(blocks[4].state.speculative_evicted_ns==first_eviction);
 now+=10*NSEC_PER_MSEC;grant=0;run();run();assert(feedback_quick==1);
 // An expired demand consumes the token and cannot be revived by another signal.
 reset();blocks[4].state.speculative_cohort=42;
 blocks[4].state.speculative_evicted_ns=now-20*NSEC_PER_MSEC-1;
 grant=0;run();run();assert(!feedback_quick && !blocks[4].state.speculative_cohort);
 // Sparse learned regions 1 and 4: move trigger and sibling, then region4,
 // never transfer the intervening virtual regions or cross the 64MiB tile.
 reset();range.count=64;
 for(size_t i=0;i<64;++i){blocks[i].managed_range=&range;blocks[i].start=i*UVM_VA_BLOCK_SIZE;range.blocks[i]=&blocks[i];}
 trigger_slot=2;grant=8*UVM_SIZE_1MB;
 uvm_cpu_block_policy_prefetch_on_signal(&blocks[2],&gpu,&mm,(1u<<1)|(1u<<4));
 assert(requested==8*UVM_SIZE_1MB && calls==4 && visited[0]==2 && visited[1]==3 && visited[2]==8 && visited[3]==9);
 assert(transferred==grant && completed_success && !reserved);
 // Entire learned tile remains bounded by a smaller concurrent budget grant.
 reset();range.count=64;
 for(size_t i=0;i<64;++i){blocks[i].managed_range=&range;blocks[i].start=i*UVM_VA_BLOCK_SIZE;range.blocks[i]=&blocks[i];}
 trigger_slot=31;grant=6*UVM_SIZE_1MB;
 uvm_cpu_block_policy_prefetch_on_signal(&blocks[31],&gpu,&mm,0xffff);
 assert(requested==64*UVM_SIZE_1MB && calls==3 && visited[0]==31 && visited[1]==30 && visited[2]==0);
 // Partial first block at an unaligned range start: never visit the prior half.
 reset();range.start=3*UVM_VA_BLOCK_SIZE+4096;range.count=4;trigger_slot=0;grant=6*UVM_SIZE_1MB;
 for(size_t i=0;i<4;++i){blocks[i].start=(3+i)*UVM_VA_BLOCK_SIZE;range.blocks[i]=&blocks[i];}
 blocks[0].start=range.start;
 uvm_cpu_block_policy_prefetch_on_signal(&blocks[0],&gpu,&mm,(1u<<1)|(1u<<2));
 assert(calls==3 && visited[0]==0 && visited[1]==1 && visited[2]==2 && requested==grant);
 // Allocation ending in the first half of a region must not invent its sibling.
 reset();range.count=3;trigger_slot=2;grant=2*UVM_SIZE_1MB;
 uvm_cpu_block_policy_prefetch_on_signal(&blocks[2],&gpu,&mm,(1u<<1));
 assert(calls==1 && visited[0]==2 && requested==grant);
 // A tile boundary must clip a long allocation beginning before that tile.
 reset();range.count=64;
 for(size_t i=0;i<64;++i){blocks[i].managed_range=&range;blocks[i].start=i*UVM_VA_BLOCK_SIZE;range.blocks[i]=&blocks[i];}
 trigger_slot=32;grant=64*UVM_SIZE_1MB;
 uvm_cpu_block_policy_prefetch_on_signal(&blocks[32],&gpu,&mm,0xffff);
 assert(calls==32 && visited[0]==32 && visited[1]==33);
 for(int i=0;i<calls;++i)assert(visited[i]>=32 && visited[i]<64);
 // Refault admission copies only additional neighbors. The demand trigger
 // itself is never copied speculatively, and the 64MiB tile bounds neighbors.
 reset();range.count=64;
 for(size_t i=0;i<64;++i){blocks[i].managed_range=&range;blocks[i].start=i*UVM_VA_BLOCK_SIZE;range.blocks[i]=&blocks[i];}
 trigger_slot=31; refault_enabled=true; refault_neighbor_count=2; grant=4*UVM_SIZE_1MB;
 uvm_cpu_block_policy_prefetch_on_signal(&blocks[31],&gpu,&mm,0);
 assert(requested==4*UVM_SIZE_1MB && calls==2 && visited[0]==30 && visited[1]==29);
 assert(transferred==grant && completed_success && !reserved);
 assert(!uvm_cpu_preferred_trigger_prefetched_bytes && (NvU64)uvm_cpu_preferred_neighbor_prefetched_bytes==grant);
 // At the beginning of a tile only forward neighbors are legal.
 reset();range.count=64;
 for(size_t i=0;i<64;++i){blocks[i].managed_range=&range;blocks[i].start=i*UVM_VA_BLOCK_SIZE;range.blocks[i]=&blocks[i];}
 trigger_slot=0; refault_enabled=true; refault_neighbor_count=4; grant=8*UVM_SIZE_1MB;
 uvm_cpu_block_policy_prefetch_on_signal(&blocks[0],&gpu,&mm,0);
 assert(calls==4 && visited[0]==1 && visited[1]==2 && visited[2]==3 && visited[3]==4);
 // An empty admission makes no reservation and performs no speculative copy.
 reset();range.count=64;
 for(size_t i=0;i<64;++i){blocks[i].managed_range=&range;blocks[i].start=i*UVM_VA_BLOCK_SIZE;range.blocks[i]=&blocks[i];}
 trigger_slot=32; refault_enabled=true; refault_neighbor_count=0; grant=8*UVM_SIZE_1MB;
 uvm_cpu_block_policy_prefetch_on_signal(&blocks[32],&gpu,&mm,0);
 assert(!begin_calls && !end_calls && !reserved && !calls);
 puts("Production prefetch walker: bounded copies, trigger order, retry, balanced failures, waits and churn feedback passed.");
 return 0;
}
'''


def main():
    source = (ROOT / 'kernel-open/nvidia-uvm/uvm_cpu_block_policy.c').read_text()
    code = PRELUDE + function(source, 'uvm_cpu_block_policy_prefetch_on_signal')
    code += function(source, 'uvm_cpu_block_policy_rearm_on_eviction') + CASES
    with tempfile.TemporaryDirectory(prefix='speculative-prefetch-') as directory:
        c = Path(directory) / 'test.c'
        binary = Path(directory) / 'test'
        c.write_text(code)
        subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', '-g', str(c), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
