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
#define NV_OK 0
#define UVM_SIZE_1MB (1ULL << 20)
#define UVM_VA_BLOCK_SIZE (2ULL << 20)
#define NSEC_PER_MSEC 1000000ULL
#define UVM_ASSERT(x) assert(x)
#define UVM_ALIGN_DOWN(x,a) ((x) & ~((NvU64)(a)-1))
#define ARRAY_SIZE(x) (sizeof(x)/sizeof((x)[0]))
#define READ_ONCE(x) (x)
#define min(a,b) ((a)<(b)?(a):(b))
#define UVM_DBG_PRINT(...) do {} while(0)
struct mm_struct { int unused; };
typedef struct { int lock; } uvm_va_space_t;
typedef struct { NvU64 speculative_prefetch_ns, speculative_evicted_ns, speculative_cohort; } uvm_va_block_gpu_state_t;
typedef struct { int unused; } uvm_pmm_gpu_t;
typedef struct { NvU64 bytes; NvU64 generation; } uvm_prefetch_reservation_t;
typedef struct { int id; uvm_pmm_gpu_t pmm; } uvm_gpu_t;
typedef struct uvm_va_range_managed_struct uvm_va_range_managed_t;
typedef struct {
 uvm_va_range_managed_t *managed_range; NvU64 start; int lock;
 unsigned gpu_prefetch_started, gpu_prefetch_retry; uvm_va_block_gpu_state_t state;
} uvm_va_block_t;
struct uvm_va_range_managed_struct {
 struct { uvm_va_space_t *va_space; } va_range;
 NvU64 start; size_t count; uvm_va_block_t *blocks[64];
};
typedef struct { int unused; } uvm_va_block_context_t;
typedef struct { int unused; } uvm_tracker_t;
#define UVM_TRACKER_INIT() {0}
static NvU64 now = 1000000000ULL;
static unsigned uvm_cpu_preferred_prefetch_chunk_mb;
static long long uvm_cpu_preferred_prefetch_capacity_stops, uvm_cpu_preferred_range_prefetches;
static long long uvm_cpu_preferred_prefetch_retries, uvm_cpu_preferred_prefetched_bytes;
static void atomic64_inc(long long *p) {++*p;}
static void atomic64_add(NvU64 n,long long *p) {*p+=(long long)n;}
static bool enabled, supported, alloc_fail, trigger_resident;
static NvU64 grant, requested, reserved, transferred;
static bool completed_success;
static NvU64 reservation_generation = 17;
static int calls, begin_calls, end_calls, waits, capacity_at, error_at, wait_error;
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
static NvU64 uvm_pmm_gpu_prefetch_feedback_window_ns(void) {return 20*NSEC_PER_MSEC;}
static void uvm_pmm_gpu_prefetch_pressure_pause(uvm_pmm_gpu_t *p) {(void)p;++pressure_pauses;}
static void uvm_pmm_gpu_prefetch_return(uvm_pmm_gpu_t *p,NvU64 cohort,NvU64 t) {(void)p;assert(t==now);if (cohort) {assert(cohort==42);++feedback_quick;}}
static void uvm_pmm_gpu_prefetch_allocation_failure(uvm_pmm_gpu_t *p) {(void)p;++feedback_capacity;}
static bool uvm_processor_mask_test(unsigned *p,int id) {(void)id;return *p;}
static bool uvm_processor_mask_test_and_clear_atomic(unsigned *p,int id) {(void)id;bool old=*p;*p=0;return old;}
static bool uvm_processor_mask_test_and_set_atomic(unsigned *p,int id) {(void)id;bool old=*p;*p=1;return old;}
static void uvm_processor_mask_clear_atomic(unsigned *p,int id) {(void)id;*p=0;}
static void uvm_processor_mask_set_atomic(unsigned *p,int id) {(void)id;*p=1;}
static uvm_va_block_context_t *uvm_va_block_context_alloc(struct mm_struct *m) {(void)m;return alloc_fail?NULL:&context;}
static void uvm_va_block_context_free(uvm_va_block_context_t *p) {assert(p==&context);}
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
 calls=begin_calls=end_calls=waits=capacity_at=error_at=wait_error=0;
 watermark_at=-1;watermark_calls=feedback_quick=feedback_capacity=pressure_pauses=0;
 trigger_slot=4;
 uvm_cpu_preferred_prefetch_chunk_mb=64;
 uvm_cpu_preferred_prefetch_capacity_stops=uvm_cpu_preferred_range_prefetches=0;
 uvm_cpu_preferred_prefetch_retries=uvm_cpu_preferred_prefetched_bytes=0;
}
static void run(void){uvm_cpu_block_policy_prefetch_on_signal(&blocks[trigger_slot],&gpu,&mm,0);}
int main(void) {
 // Actual production loop: range clamps request; grant truncates it; trigger is first.
 reset();run();assert(requested==16*UVM_SIZE_1MB && calls==2 && visited[0]==4 && visited[1]==3);
 assert(end_calls==1 && completed_success && transferred==grant && !reserved);
 reset();grant=8*UVM_SIZE_1MB;run();assert(calls==4 && visited[0]==4 && visited[1]==2 && visited[2]==3 && visited[3]==5);
 reset();grant=8*UVM_SIZE_1MB;trigger_slot=7;run();assert(calls==4 && visited[0]==7 && visited[1]==4 && visited[2]==5 && visited[3]==6);
 // A resident triggering block does not prevent neighboring transfers.
 reset();trigger_resident=true;run();assert(calls==2 && transferred==UVM_VA_BLOCK_SIZE);
 // Sparse neighbor blocks consume no bytes and do not instantiate new blocks.
 reset();range.blocks[3]=NULL;run();assert(calls==1 && transferred==UVM_VA_BLOCK_SIZE && end_calls==1);
 // Transient denial retries, permanent disable never admits or sets retry.
 reset();grant=0;run();assert(!calls && !end_calls && blocks[4].gpu_prefetch_retry);
 reset();enabled=false;run();assert(!begin_calls && !calls && !waits && !blocks[4].gpu_prefetch_retry);
 reset();uvm_cpu_preferred_prefetch_chunk_mb=0;run();assert(!begin_calls && !calls);
 // Every admitted failure waits before releasing the reservation.
 reset();error_at=1;run();assert(calls==1 && end_calls==1 && !completed_success && !reserved);
 reset();capacity_at=2;run();assert(calls==2 && end_calls==1 && feedback_capacity==1 && !completed_success && !pressure_pauses && blocks[4].gpu_prefetch_retry);
 reset();watermark_at=0;run();assert(!calls && end_calls==1 && completed_success && !feedback_capacity && pressure_pauses==1);
 reset();wait_error=9;run();assert(end_calls==1 && !completed_success && !reserved);
 // Context allocation failure cannot leak an admission or change retry state.
 reset();alloc_fail=true;run();assert(!begin_calls && !end_calls && !blocks[4].gpu_prefetch_started);
 // One-shot callback avoids repeat copies; capacity-retry can be claimed once.
 reset();run();run();assert(begin_calls==1 && calls==2 && end_calls==1 && !reserved);
 reset();grant=0;run();grant=4*UVM_SIZE_1MB;waits=0;run();assert(end_calls==1 && calls==2 && uvm_cpu_preferred_prefetch_retries==1);
 // Actual eviction hook and actual demand callback: quick return is consumed once.
 reset();blocks[4].state.speculative_cohort=42;blocks[4].state.speculative_prefetch_ns=now-10;blocks[4].lock=1;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu);blocks[4].lock=0;
 assert(blocks[4].state.speculative_evicted_ns==now && !blocks[4].state.speculative_prefetch_ns);
 grant=0;run();run();assert(feedback_quick==1 && !blocks[4].state.speculative_evicted_ns && !blocks[4].state.speculative_cohort);
 // Expired returns and expired prefetch-to-eviction intervals cannot qualify.
 reset();blocks[4].state.speculative_evicted_ns=now-21*NSEC_PER_MSEC;grant=0;run();assert(!feedback_quick);
 reset();blocks[4].state.speculative_prefetch_ns=now-21*NSEC_PER_MSEC;blocks[4].lock=1;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu);blocks[4].lock=0;grant=0;run();assert(!feedback_quick);
 // Boundary is inclusive for each interval; a second partial eviction cannot refresh it.
 reset();blocks[4].state.speculative_cohort=42;
 blocks[4].state.speculative_prefetch_ns=now-20*NSEC_PER_MSEC;blocks[4].lock=1;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu);blocks[4].lock=0;
 NvU64 first_eviction=blocks[4].state.speculative_evicted_ns;
 assert(first_eviction==now);
 now+=10*NSEC_PER_MSEC;blocks[4].lock=1;
 uvm_cpu_block_policy_rearm_on_eviction(&blocks[4],&gpu);blocks[4].lock=0;
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
