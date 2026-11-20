#!/usr/bin/env python3
"""Exercise production admission, budget, bounded worker and completion together."""
from pathlib import Path
import subprocess, tempfile
from test_demand_first_eviction import function

ROOT = Path(__file__).parents[2]
PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
typedef uint64_t NvU64; typedef unsigned NvU32;
#define UVM_PAGE_SIZE_64K 65536ULL
#define UVM_CHUNK_SIZE_MAX (2ULL<<20)
#define UVM_SIZE_1MB (1ULL<<20)
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define clamp(a,b,c) min(max(a,b),c)
#define DIV_ROUND_UP_ULL(a,b) (((a)+(b)-1)/(b))
#define READ_ONCE(a) (a)
#define UVM_ASSERT(x) assert(x)
#define system_wq 0
/* REAL BUDGET */
typedef struct {NvU64 numFreePages64k;} pma_t;
typedef struct {struct {struct {bool enabled;} numa;} mem_info;} uvm_gpu_t;
typedef struct {struct {uvm_prefetch_budget_t budget;unsigned prefetch_count,replacement_roots;NvU64 replacement_requests,replacement_freed_roots;bool recovering_watermark,active,stopping;int work;} eager_eviction;pma_t *pma_stats;int list_lock;struct {NvU64 cached_free_root_count;} root_chunks;} uvm_pmm_gpu_t;
static uvm_gpu_t gpu;static bool supported=true;static unsigned usage=99,scheduled;static unsigned batch_calls,max_batch=16;static bool uvm_perf_prefetch_replacement_enable=true;static unsigned uvm_perf_eager_eviction_batch_count=16,uvm_perf_eager_eviction_high_watermark=99;
static uvm_gpu_t *uvm_pmm_to_gpu(uvm_pmm_gpu_t*p){(void)p;return &gpu;} static bool eager_eviction_supported(uvm_pmm_gpu_t*p){(void)p;return supported;} static unsigned eager_eviction_usage_percent(uvm_pmm_gpu_t*p){(void)p;return usage;}
static void uvm_spin_lock(int*p){assert(!*p);*p=1;}static void uvm_spin_unlock(int*p){assert(*p);*p=0;}static void uvm_assert_spinlock_locked(int*p){assert(*p);}
static uvm_prefetch_budget_config_t speculative_prefetch_config(void){return (uvm_prefetch_budget_config_t){.demand_reserve_bytes=64ULL<<20,.limit_bytes=32ULL<<20,.observation_ns=20000000,.return_percent=50,.min_blocks=32,.bad_batches=2,.good_batches=4};} static unsigned long NV_GETTIME(void){return 1;}
static unsigned long last_delay;
static void eager_eviction_schedule(uvm_pmm_gpu_t*p,unsigned long d){assert(!p->list_lock);last_delay=d;++scheduled;}
struct work_struct {int unused;};
static uvm_pmm_gpu_t *work_pmm;
#define to_delayed_work(w) (w)
#define container_of(ptr,type,member) ((void)(ptr),work_pmm)
static unsigned uvm_perf_eager_eviction_low_watermark=90, uvm_perf_eager_eviction_interval_ms=25;
static unsigned long msecs_to_jiffies(unsigned ms) {return ms;}
static unsigned prepares;
static void vab_prepare_roots(uvm_pmm_gpu_t *p) {assert(!p->list_lock);++prepares;}
static unsigned expedited;
static void mod_delayed_work(int q,int*w,unsigned long d){assert(!q&&w&&!d);++expedited;}
static unsigned eager_eviction_batch(uvm_pmm_gpu_t *p,unsigned limit,unsigned low) {
 assert(!p->list_lock && p->eager_eviction.active && !p->eager_eviction.prefetch_count);
 assert(limit && limit<=16 && !low);++batch_calls;unsigned freed=min(limit,max_batch);
 p->pma_stats->numFreePages64k+=(NvU64)freed*UVM_CHUNK_SIZE_MAX/UVM_PAGE_SIZE_64K;
 return freed;
}
'''
CASES = r'''
static uvm_pmm_gpu_t p;
static pma_t stats;
static uvm_prefetch_reservation_t r;
static void reset(void) {
 memset(&p,0,sizeof(p));memset(&stats,0,sizeof(stats));memset(&gpu,0,sizeof(gpu));
 p.pma_stats=&stats;supported=true;usage=99;scheduled=expedited=batch_calls=prepares=0;last_delay=0;max_batch=16;
 uvm_perf_prefetch_replacement_enable=true;
}
static NvU64 begin(NvU64 bytes) {return uvm_pmm_gpu_eager_eviction_prefetch_begin(&p,bytes,&r);}
int main(void) {
 // No demand or invalid request cannot start reclaim.
 reset();assert(!replacement_eviction_work(&p,16) && !batch_calls);
 assert(!begin(0) && !p.eager_eviction.replacement_roots && !expedited);
 // The full 96 MiB shortage survives three capped passes, without new reports.
 reset();assert(!begin(UVM_CHUNK_SIZE_MAX));
 assert(p.eager_eviction.replacement_roots==48 && expedited==1);
 for(int i=0;i<100;++i) assert(!begin(UVM_CHUNK_SIZE_MAX));
 assert(p.eager_eviction.replacement_roots==48 && expedited==1);
 assert(replacement_eviction_work(&p,16) && p.eager_eviction.replacement_roots==32);
 assert(replacement_eviction_work(&p,16) && p.eager_eviction.replacement_roots==16);
 assert(!replacement_eviction_work(&p,16) && !p.eager_eviction.replacement_roots);
 assert(batch_calls==3 && p.eager_eviction.replacement_freed_roots==48);
 assert(stats.numFreePages64k*UVM_PAGE_SIZE_64K==(96ULL<<20));
 assert(begin(UVM_CHUNK_SIZE_MAX)==UVM_CHUNK_SIZE_MAX);
 assert(p.eager_eviction.prefetch_count==1 && p.eager_eviction.budget.reserved_bytes==UVM_CHUNK_SIZE_MAX);
 stats.numFreePages64k-=UVM_CHUNK_SIZE_MAX/UVM_PAGE_SIZE_64K;
 uvm_pmm_gpu_eager_eviction_prefetch_end(&p,&r,UVM_CHUNK_SIZE_MAX,true);
 assert(!p.eager_eviction.prefetch_count && !p.eager_eviction.budget.reserved_bytes);
 assert(!replacement_eviction_work(&p,16) && batch_calls==3);
 // Policy OFF and cap denials cannot reclaim, even with no free space.
 reset();p.eager_eviction.budget.initialized=true;p.eager_eviction.budget.window_bytes=0;
 assert(!begin(UVM_CHUNK_SIZE_MAX) && !p.eager_eviction.replacement_roots && !expedited);
 reset();p.eager_eviction.budget.initialized=true;p.eager_eviction.budget.window_bytes=32ULL<<20;
 p.eager_eviction.budget.reserved_bytes=32ULL<<20;p.eager_eviction.prefetch_count=1;
 assert(!begin(UVM_CHUNK_SIZE_MAX) && !p.eager_eviction.replacement_roots && !expedited);
 // Cached roots cancel stale demand and permit real budget admission.
 reset();assert(!begin(UVM_CHUNK_SIZE_MAX));p.root_chunks.cached_free_root_count=48;
 assert(!replacement_eviction_work(&p,16) && !batch_calls && !p.eager_eviction.replacement_roots);
 assert(begin(UVM_CHUNK_SIZE_MAX)==UVM_CHUNK_SIZE_MAX);
 uvm_pmm_gpu_eager_eviction_prefetch_end(&p,&r,0,false);
 assert(!p.eager_eviction.budget.reserved_bytes);
 reset();p.root_chunks.cached_free_root_count=48;gpu.mem_info.numa.enabled=true;
 assert(!begin(UVM_CHUNK_SIZE_MAX) && p.eager_eviction.replacement_roots==48);
 // Cap-only denial cannot request replacement or exceed inflight bytes.
 reset();stats.numFreePages64k=(1024ULL<<20)/UVM_PAGE_SIZE_64K;
 assert(begin(32ULL<<20)==(32ULL<<20));uvm_prefetch_reservation_t held=r;
 assert(!begin(UVM_CHUNK_SIZE_MAX) && !p.eager_eviction.replacement_roots && !expedited);
 assert(p.eager_eviction.budget.reserved_bytes==(32ULL<<20));
 uvm_pmm_gpu_eager_eviction_prefetch_end(&p,&held,0,true);
 // Native freeing shrinks the shortfall before the worker touches a root.
 reset();assert(!begin(UVM_CHUNK_SIZE_MAX));stats.numFreePages64k=(94ULL<<20)/UVM_PAGE_SIZE_64K;
 assert(replacement_eviction_work(&p,16) && p.eager_eviction.replacement_freed_roots==1);
 assert(!replacement_eviction_work(&p,16) && !p.eager_eviction.replacement_roots);
 // No candidate backs off with pending demand; duplicates do not re-kick.
 reset();assert(!begin(UVM_CHUNK_SIZE_MAX));max_batch=0;
 assert(!replacement_eviction_work(&p,16) && p.eager_eviction.replacement_roots==48);
 assert(!p.eager_eviction.active && batch_calls==1);
 assert(!begin(UVM_CHUNK_SIZE_MAX) && expedited==1);
 // Inflight populations and active reclaim exclude one another.
 reset();p.eager_eviction.replacement_roots=48;p.eager_eviction.prefetch_count=1;
 assert(!replacement_eviction_work(&p,16) && !batch_calls);
 p.eager_eviction.prefetch_count=0;p.eager_eviction.active=true;
 assert(!begin(UVM_CHUNK_SIZE_MAX) && !scheduled && !expedited);
 // Last population completion wakes pending refill.
 reset();stats.numFreePages64k=(96ULL<<20)/UVM_PAGE_SIZE_64K;
 assert(begin(UVM_CHUNK_SIZE_MAX)==UVM_CHUNK_SIZE_MAX);p.eager_eviction.replacement_roots=4;
 uvm_pmm_gpu_eager_eviction_prefetch_end(&p,&r,0,true);assert(expedited==1);
 // Actual workqueue wrapper yields immediately only on productive remainder.
 reset();assert(!begin(UVM_CHUNK_SIZE_MAX));work_pmm=&p;struct work_struct work={0};
 eager_eviction_work_impl(&work);assert(p.eager_eviction.replacement_roots==32 && last_delay==0);
 eager_eviction_work_impl(&work);assert(p.eager_eviction.replacement_roots==16 && last_delay==0);
 eager_eviction_work_impl(&work);assert(!p.eager_eviction.replacement_roots && last_delay==25);
 assert(scheduled==3 && batch_calls==3);
 reset();assert(!begin(UVM_CHUNK_SIZE_MAX));max_batch=0;
 eager_eviction_work_impl(&work);assert(last_delay==25 && p.eager_eviction.replacement_roots==48);
 unsigned previous_schedules=scheduled;p.eager_eviction.stopping=true;
 eager_eviction_work_impl(&work);assert(scheduled==previous_schedules && batch_calls==1);
 // Flag-off preserves watermark pause and does not credit PMM cache.
 reset();uvm_perf_prefetch_replacement_enable=false;p.root_chunks.cached_free_root_count=48;
 assert(!begin(UVM_CHUNK_SIZE_MAX) && p.eager_eviction.recovering_watermark);
 assert(scheduled==1 && !expedited && !p.eager_eviction.replacement_roots);
 puts("Production bounded refill: whole shortage, worker, retry, cache, cap, failure and legacy passed.");
}
'''

def main():
    source=(ROOT/'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text(); budget=(ROOT/'kernel-open/nvidia-uvm/uvm_prefetch_budget.h').read_text().replace('#include "nvtypes.h"',''); code=PRELUDE.replace('/* REAL BUDGET */',budget)
    for name in ('prefetch_cached_free_bytes_locked','prefetch_free_bytes_locked','replacement_refill_bytes','eager_eviction_prefetch_blocked_locked','uvm_pmm_gpu_eager_eviction_prefetch_begin','uvm_pmm_gpu_eager_eviction_prefetch_end','replacement_eviction_work','eager_eviction_work_impl'):
        code += function(source,name)
    code += CASES
    with tempfile.TemporaryDirectory() as d:
        c=Path(d)/'test.c';b=Path(d)/'test';c.write_text(code)
        subprocess.run(['cc','-std=c11','-O1','-Wall','-Wextra','-Werror','-Wno-unused-function','-fsanitize=address,undefined',str(c),'-o',str(b)],check=True);subprocess.run([str(b)],check=True)
if __name__=='__main__':main()
