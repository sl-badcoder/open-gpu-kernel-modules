#!/usr/bin/env python3
"""Compile the production reservation/reclaim handoff and test its boundaries."""
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
#define UVM_PAGE_SIZE_64K 65536ULL
#define UVM_CHUNK_SIZE_MAX (2ULL<<20)
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define clamp(a,b,c) min(max(a,b),c)
#define DIV_ROUND_UP_ULL(a,b) (((a)+(b)-1)/(b))
#define READ_ONCE(a) (a)
#define UVM_ASSERT(a) assert(a)
#define system_wq 0
static unsigned long NV_GETTIME(void) {return 42;}
typedef struct {NvU64 demand_reserve_bytes;} uvm_prefetch_budget_config_t;
typedef struct {NvU64 bytes;} uvm_prefetch_reservation_t;
typedef struct {NvU64 reserved_bytes, denials, pauses;} budget_t;
typedef struct {NvU64 numFreePages64k;} pma_t;
typedef struct {struct {budget_t budget; unsigned prefetch_count, replacement_roots;
 NvU64 replacement_requests; bool recovering_watermark, active, stopping; int work;} eager_eviction;
 pma_t *pma_stats; int list_lock;} uvm_pmm_gpu_t;
static bool uvm_perf_prefetch_replacement_enable, supported;
static unsigned uvm_perf_eager_eviction_batch_count=16;
static unsigned uvm_perf_eager_eviction_high_watermark=99, usage;
static unsigned scheduled, resumed;
static NvU64 grant;
static uvm_prefetch_budget_config_t config;
static bool eager_eviction_supported(uvm_pmm_gpu_t *p) {(void)p; return supported;}
static unsigned eager_eviction_usage_percent(uvm_pmm_gpu_t *p) {(void)p; return usage;}
static void uvm_assert_spinlock_locked(int *p) {assert(*p);}
static void uvm_spin_lock(int *p) {assert(!*p);*p=1;}
static void uvm_spin_unlock(int *p) {assert(*p);*p=0;}
static uvm_prefetch_budget_config_t speculative_prefetch_config(void) {return config;}
static void eager_eviction_schedule(uvm_pmm_gpu_t *p,unsigned delay) {assert(!p->list_lock&&!delay);++scheduled;}
static void mod_delayed_work(int q,int *w,unsigned delay) {(void)q;(void)w;assert(!delay);++resumed;}
static void uvm_prefetch_budget_pressure_pause(budget_t *b) {++b->pauses;}
static NvU64 uvm_prefetch_budget_reserve(budget_t *b,const uvm_prefetch_budget_config_t *c,
 NvU64 requested,NvU64 free_bytes,unsigned long now,uvm_prefetch_reservation_t *r)
{(void)c;(void)free_bytes;assert(now==42);assert(grant<=requested);r->bytes=grant;b->reserved_bytes+=grant;return grant;}
static void uvm_prefetch_budget_complete(budget_t *b,const uvm_prefetch_budget_config_t *c,
 const uvm_prefetch_reservation_t *r,NvU64 bytes,bool success,unsigned long now)
{(void)c;(void)success;assert(now==42&&bytes<=r->bytes&&b->reserved_bytes>=r->bytes);b->reserved_bytes-=r->bytes;}
'''
CASES = r'''
static uvm_pmm_gpu_t p;static pma_t stats;static uvm_prefetch_reservation_t r;
static void reset(void) {memset(&p,0,sizeof(p));memset(&stats,0,sizeof(stats));p.pma_stats=&stats;
 uvm_perf_prefetch_replacement_enable=true;supported=true;usage=99;grant=0;scheduled=resumed=0;
 config.demand_reserve_bytes=64ULL<<20;uvm_perf_eager_eviction_batch_count=16;}
int main(void) {
 // No headroom: coalesce, cap, and wake the asynchronous worker.
 reset();assert(!uvm_pmm_gpu_eager_eviction_prefetch_begin(&p,2ULL<<20,&r));
 assert(p.eager_eviction.replacement_roots==16&&p.eager_eviction.replacement_requests==1&&scheduled==1);
 for(unsigned i=0;i<100;++i)uvm_pmm_gpu_eager_eviction_prefetch_begin(&p,2ULL<<20,&r);
 assert(p.eager_eviction.replacement_roots==16);
 // Byte-limit denial despite sufficient free memory never requests reclaim.
 reset();stats.numFreePages64k=2048;uvm_pmm_gpu_eager_eviction_prefetch_begin(&p,2ULL<<20,&r);
 assert(!p.eager_eviction.replacement_roots&&!scheduled);
 // Outstanding reservations are included in the headroom deficit.
 reset();stats.numFreePages64k=1056;p.eager_eviction.budget.reserved_bytes=4ULL<<20;
 uvm_pmm_gpu_eager_eviction_prefetch_begin(&p,2ULL<<20,&r);assert(p.eager_eviction.replacement_roots==2);
 // Successful reservation permits population even above the old watermark.
 reset();grant=2ULL<<20;stats.numFreePages64k=2048;
 assert(uvm_pmm_gpu_eager_eviction_prefetch_begin(&p,grant,&r)==grant&&p.eager_eviction.prefetch_count==1);
 p.eager_eviction.replacement_roots=1;uvm_pmm_gpu_eager_eviction_prefetch_end(&p,&r,grant,true);
 assert(!p.eager_eviction.prefetch_count&&!p.eager_eviction.budget.reserved_bytes&&resumed==1);
 // Active reclaim excludes population; do not enqueue repeated requests.
 reset();p.eager_eviction.active=true;grant=2ULL<<20;
 assert(!uvm_pmm_gpu_eager_eviction_prefetch_begin(&p,grant,&r));
 assert(!p.eager_eviction.prefetch_count&&!p.eager_eviction.replacement_roots&&!scheduled);
 // No eager worker means no replacement request, preserving the disabled case.
 reset();supported=false;uvm_pmm_gpu_eager_eviction_prefetch_begin(&p,2ULL<<20,&r);
 assert(!p.eager_eviction.replacement_roots&&!scheduled);
 // Legacy mode still blocks on watermark and wakes recovery.
 reset();uvm_perf_prefetch_replacement_enable=false;grant=2ULL<<20;
 assert(!uvm_pmm_gpu_eager_eviction_prefetch_begin(&p,grant,&r));
 assert(p.eager_eviction.recovering_watermark&&scheduled==1&&!p.eager_eviction.replacement_roots);
 assert(!p.list_lock);
 puts("Production replacement requests: bounded coalescing, reservation handoff, exclusion and legacy fallback passed.");
}
'''

def main():
    source = (ROOT / 'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text()
    code = PRELUDE
    for name in ('eager_eviction_prefetch_blocked_locked',
                 'uvm_pmm_gpu_eager_eviction_prefetch_begin',
                 'uvm_pmm_gpu_eager_eviction_prefetch_end'):
        code += function(source, name)
    code += CASES
    with tempfile.TemporaryDirectory(prefix='replacement-requests-') as directory:
        path=Path(directory)/'test.c'; binary=Path(directory)/'test'
        path.write_text(code)
        subprocess.run(['cc','-std=c11','-O1','-Wall','-Wextra','-Werror',
                        '-fsanitize=address,undefined',str(path),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)

if __name__ == '__main__':
    main()
