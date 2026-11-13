#!/usr/bin/env python3
"""Exercise actual speculative population accounting with mocked residency I/O."""
from pathlib import Path
import subprocess
import tempfile
from test_demand_first_eviction import function

ROOT = Path(__file__).parents[2]
PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef uint64_t NvU64;
typedef int NV_STATUS;
typedef NvU64 uvm_page_mask_t;
typedef int uvm_tracker_t;
#define UVM_ASSERT assert
#define NV_OK 0
#define NV_ERR_MORE_PROCESSING_REQUIRED 1
#define PAGE_SIZE 4096
#define UVM_ID_CPU 0
#define NUMA_NO_NODE (-1)
#define UVM_MAKE_RESIDENT_CAUSE_PREFETCH 3
#define UVM_PROT_READ_WRITE_ATOMIC 4
#define UvmEventMapRemoteCauseInvalid 0
struct mm_struct { int unused; };
typedef struct { bool allow_eviction, no_eviction_allocation_failed; } uvm_va_block_retry_t;
typedef struct { struct { uvm_page_mask_t pages_changed_residency; unsigned all_involved_processors; } make_resident; } uvm_va_block_context_t;
typedef struct { NvU64 speculative_prefetch_ns, speculative_evicted_ns, speculative_cohort; } uvm_va_block_gpu_state_t;
typedef struct { int lock; uvm_tracker_t tracker; uvm_va_block_gpu_state_t state; } uvm_va_block_t;
typedef struct { unsigned unused; } uvm_pmm_gpu_t;
typedef struct { int id; uvm_pmm_gpu_t pmm; } uvm_gpu_t;
static uvm_page_mask_t cpu, resident, thrashing, changed;
static bool has_cpu, fail_allocation;
static NV_STATUS resident_status, map_status, tracker_status;
static unsigned records, maps, retries, dependencies;
static NvU64 recorded_now, recorded_generation;
static void uvm_mutex_lock(int *l) { assert(!*l); *l=1; }
static void uvm_mutex_unlock(int *l) { assert(*l); *l=0; }
static const uvm_page_mask_t *uvm_va_block_resident_mask_get(uvm_va_block_t *b,int id,int node)
{ assert(b->lock && node==-1); return id==0 ? (has_cpu?&cpu:NULL) : &resident; }
static const uvm_page_mask_t *uvm_perf_thrashing_get_thrashing_pages(uvm_va_block_t *b)
{ assert(b->lock); return &thrashing; }
static void uvm_page_mask_copy(uvm_page_mask_t *d,const uvm_page_mask_t *s) { *d=*s; }
static void uvm_page_mask_andnot(uvm_page_mask_t *d,const uvm_page_mask_t *a,const uvm_page_mask_t *b) { *d=*a & ~*b; }
static bool uvm_page_mask_empty(const uvm_page_mask_t *m) { return !*m; }
static void uvm_page_mask_zero(uvm_page_mask_t *m) { *m=0; }
static unsigned uvm_page_mask_weight(const uvm_page_mask_t *m) { return __builtin_popcountll(*m); }
static void uvm_processor_mask_zero(unsigned *m) { *m=0; }
static void uvm_va_block_context_init(uvm_va_block_context_t *c,struct mm_struct *m) { (void)c;(void)m; }
static void uvm_va_block_retry_init(uvm_va_block_retry_t *r) { memset(r,0,sizeof(*r)); r->allow_eviction=true; }
static int uvm_va_block_region_from_block(uvm_va_block_t *b) { assert(b->lock); return 0; }
static NV_STATUS uvm_va_block_make_resident(uvm_va_block_t *b,uvm_va_block_retry_t *r,
 uvm_va_block_context_t *c,int id,int region,const uvm_page_mask_t *mask,void *unused,int cause)
{ assert(b->lock && !r->allow_eviction && id==1 && !region && !unused && cause==3);
 assert(*mask==(cpu & ~resident & ~thrashing)); c->make_resident.pages_changed_residency=changed;
 r->no_eviction_allocation_failed=fail_allocation; return resident_status; }
static uvm_va_block_gpu_state_t *uvm_va_block_gpu_state_get(uvm_va_block_t *b,int id)
{ assert(b->lock && id==1); return &b->state; }
static NvU64 NV_GETTIME(void) { return 123456789; }
static NvU64 uvm_pmm_gpu_prefetch_record(uvm_pmm_gpu_t *p,NvU64 now,NvU64 generation)
{ (void)p; ++records; recorded_now=now; recorded_generation=generation; return 42; }
static NV_STATUS uvm_va_block_map(uvm_va_block_t *b,uvm_va_block_context_t *c,int id,
 int region,const uvm_page_mask_t *mask,int prot,int cause,uvm_tracker_t *tracker)
{ (void)c; assert(b->lock && id==1 && !region && *mask && prot==4 && !cause && tracker==&b->tracker);
 ++maps; return map_status; }
static NV_STATUS uvm_tracker_add_tracker_safe(uvm_tracker_t *dst,uvm_tracker_t *src)
{ (void)dst;(void)src; ++dependencies; return tracker_status; }
static void uvm_va_block_retry_deinit(uvm_va_block_retry_t *r,uvm_va_block_t *b)
{ (void)r; assert(!b->lock); ++retries; }
'''
CASES = r'''
static uvm_va_block_t block;
static uvm_gpu_t gpu={.id=1};
static uvm_va_block_context_t context;
static uvm_tracker_t tracker;
static bool capacity;
static NvU64 moved;
static void reset(void)
{
 memset(&block,0,sizeof(block)); memset(&context,0,sizeof(context));
 has_cpu=true; fail_allocation=false; cpu=7; resident=2; thrashing=4; changed=1;
 resident_status=map_status=tracker_status=0;
 records=maps=retries=dependencies=0; moved=recorded_now=recorded_generation=0;
}
static NV_STATUS run(bool neighbor)
{ return prefetch_block(&block,&gpu,NULL,&context,&tracker,&moved,17,&capacity,neighbor); }
int main(void)
{
 // One moved sparse page still counts as one neighbor population episode.
 reset(); assert(run(true)==0 && moved==PAGE_SIZE && records==1 && maps==1);
 assert(block.state.speculative_cohort==42 && block.state.speculative_prefetch_ns==123456789);
 assert(recorded_generation==17 && recorded_now==123456789 && !capacity && !block.lock);
 assert(retries==1 && dependencies==1);
 // Demand-triggering populations never enter the speculative denominator.
 reset(); assert(run(false)==0 && moved==PAGE_SIZE && !records && !block.state.speculative_cohort);
 // Empty, already resident, protected, or unchanged populations do not count.
 reset(); has_cpu=false; assert(run(true)==0 && !records && !moved && !retries);
 reset(); resident=cpu; assert(run(true)==0 && !records && !moved && !retries);
 reset(); thrashing=cpu; assert(run(true)==0 && !records && !moved && !retries);
 reset(); changed=0; assert(run(true)==0 && !records && !moved && maps==1);
 // Allocation failures do not fabricate a population, and retain no-eviction semantics.
 reset(); resident_status=7; fail_allocation=true;
 assert(run(true)==7 && capacity && !records && !moved && !maps && retries==1 && dependencies==1);
 // Errors after a successful resident transition keep already submitted accounting.
 reset(); map_status=8; assert(run(true)==8 && records==1 && moved==PAGE_SIZE && !block.lock);
 reset(); tracker_status=9; assert(run(true)==9 && records==1 && moved==PAGE_SIZE && !block.lock);
 puts("Production population accounting: moved neighbors only, sparse masks, errors and reservation generations passed.");
 return 0;
}
'''

def main():
    source = (ROOT / 'kernel-open/nvidia-uvm/uvm_cpu_block_policy.c').read_text()
    code = PRELUDE + function(source, 'prefetch_block') + CASES
    with tempfile.TemporaryDirectory(prefix='rapid-return-population-') as directory:
        c = Path(directory) / 'test.c'
        binary = Path(directory) / 'test'
        c.write_text(code)
        subprocess.run(['cc', '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=address,undefined', '-g', str(c), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

if __name__ == '__main__':
    main()
