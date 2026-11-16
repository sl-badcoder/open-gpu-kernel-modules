#!/usr/bin/env python3
"""Exercise replacement_eviction_work's bounded reclamation decisions."""
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
typedef uint64_t NvU64; typedef unsigned NvU32;
#define UVM_PAGE_SIZE_64K 65536ULL
#define UVM_CHUNK_SIZE_MAX (2ULL << 20)
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define DIV_ROUND_UP_ULL(n,d) (((n)+(d)-1)/(d))
#define READ_ONCE(x) (x)
typedef struct { NvU64 demand_reserve_bytes; } uvm_prefetch_budget_config_t;
typedef struct { NvU64 sampled_ns; bool valid; } uvm_vab_snapshot_t;
typedef struct { uvm_prefetch_replacement_state_t replacement; int list_lock; NvU64 replacement_roots, replacement_freed_roots; unsigned prefetch_count; bool active; } eager_t;
typedef struct { NvU64 numFreePages64k; } pma_stats_t;
typedef struct { eager_t eager_eviction; pma_stats_t *pma_stats; int list_lock; } uvm_pmm_gpu_t;
typedef struct { int unused; } uvm_parent_gpu_t;
typedef struct { uvm_parent_gpu_t *parent; } uvm_gpu_t;
static uvm_gpu_t mock_gpu; static pma_stats_t stats;
static uvm_prefetch_budget_config_t config = {.demand_reserve_bytes = 4ULL << 20};
static unsigned batch_calls, batch_limit, batch_low, batch_result;
static NvU64 sample_epoch = 123;
static uvm_prefetch_budget_config_t speculative_prefetch_config(void) { return config; }
static uvm_gpu_t *uvm_pmm_to_gpu(uvm_pmm_gpu_t *p) { (void)p; return &mock_gpu; }
static void uvm_gpu_vab_get_snapshot(uvm_parent_gpu_t *p, uvm_vab_snapshot_t *s) { (void)p; s->valid=true; s->sampled_ns=sample_epoch; }
static void uvm_spin_lock(int *l) { assert(!*l); *l=1; }
static void uvm_spin_unlock(int *l) { assert(*l); *l=0; }
static unsigned eager_eviction_batch(uvm_pmm_gpu_t *p, unsigned limit, unsigned low)
{ assert(p->eager_eviction.active && !p->list_lock); assert(batch_result<=limit); ++batch_calls; batch_limit=limit; batch_low=low; return batch_result; }
'''
CASES = r'''
static uvm_pmm_gpu_t p;
static void reset(void) { memset(&p,0,sizeof(p)); memset(&stats,0,sizeof(stats)); p.pma_stats=&stats; batch_calls=batch_limit=batch_low=batch_result=0; p.eager_eviction.replacement_roots=4; p.eager_eviction.replacement.request_stride=1; }
int main(void)
{
 reset(); stats.numFreePages64k=128; replacement_eviction_work(&p,8); assert(!batch_calls && p.eager_eviction.replacement_roots==0);
 reset(); p.eager_eviction.prefetch_count=1; replacement_eviction_work(&p,8); assert(!batch_calls);
 reset(); stats.numFreePages64k=0; batch_result=2; replacement_eviction_work(&p,8); assert(batch_calls==1 && batch_limit==3 && batch_low==0); assert(p.eager_eviction.replacement_freed_roots==2 && p.eager_eviction.replacement_roots==2 && !p.eager_eviction.active);
 reset(); p.eager_eviction.replacement_roots=0; replacement_eviction_work(&p,8); assert(!batch_calls);
 reset(); stats.numFreePages64k=0; p.eager_eviction.replacement_roots=16; batch_result=3; replacement_eviction_work(&p,64); assert(batch_limit==3 && p.eager_eviction.replacement_freed_roots==3 && p.eager_eviction.replacement_roots==13);
 reset(); p.eager_eviction.replacement_roots=16; config.demand_reserve_bytes=64ULL<<20; batch_result=16; replacement_eviction_work(&p,64); assert(batch_limit==16 && !p.eager_eviction.replacement_roots);
 reset(); batch_result=1; replacement_eviction_work(&p,1); assert(batch_limit==1 && p.eager_eviction.replacement_roots==3);
 reset(); batch_result=0; replacement_eviction_work(&p,8); assert(batch_calls==1 && p.eager_eviction.replacement_roots==4 && !p.eager_eviction.active && !p.eager_eviction.replacement.epoch_roots_used);
 reset(); p.eager_eviction.replacement.request_stride=4; p.eager_eviction.replacement.resident_bytes=128ULL<<20; batch_result=1;
 replacement_eviction_work(&p,8); assert(batch_limit==1 && p.eager_eviction.replacement.epoch_roots_used==1);
 replacement_eviction_work(&p,8); assert(batch_calls==1 && !p.eager_eviction.active);
 ++sample_epoch; replacement_eviction_work(&p,8); assert(batch_calls==2 && p.eager_eviction.replacement_roots==2);
 assert(!p.list_lock);
 puts("Production replacement worker: request bounds, mutual exclusion, failure and epoch quota passed.");
 return 0;
}
'''

def main():
    source = (ROOT / 'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text()
    replacement = (ROOT / 'kernel-open/nvidia-uvm/uvm_prefetch_replacement.h').read_text()
    # The extraction only needs the worker; the replacement helpers are pasted
    # before it so the quota/consume calls are compiled as production code.
    code = PRELUDE.replace('typedef struct { uvm_prefetch_replacement_state_t replacement;', 'typedef struct { uvm_prefetch_replacement_state_t replacement;')
    code = code.replace('typedef struct { NvU64 demand_reserve_bytes;', replacement + '\ntypedef struct { NvU64 demand_reserve_bytes;')
    code += function(source, 'replacement_eviction_work') + CASES
    with tempfile.TemporaryDirectory(prefix='replacement-worker-') as directory:
        c=Path(directory)/'test.c'; binary=Path(directory)/'test'; c.write_text(code)
        subprocess.run(['cc','-std=c11','-O1','-Wall','-Wextra','-Werror','-Wno-unused-function','-fsanitize=address,undefined','-g',str(c),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)

if __name__ == '__main__': main()
