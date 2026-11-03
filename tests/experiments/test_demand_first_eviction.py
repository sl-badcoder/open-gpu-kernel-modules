#!/usr/bin/env python3
"""Execute the demand-first branch's production eviction functions with mocked PMM dependencies."""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).parents[2]

def function(source, name):
    match = re.search(r"^static [^\n]*\b" + name + r"\(", source, re.M)
    if match is None:
        raise ValueError(name)
    start = match.start()
    end = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'

PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
typedef uint64_t NvU64;
typedef unsigned uvm_pmm_alloc_list_t;
typedef int NV_STATUS;
#define NV_OK 0
#define ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))
#define FREE_ROOT_CHUNK_MODE_DEFAULT 0
#define PMM_CONTEXT_DEFAULT 0
#define UVM_PMM_GPU_MEMORY_TYPE_USER 0
#define UVM_CHUNK_SIZE_MAX 2097152
#define UVM_PMM_LIST_NO_ZERO 0
#define UVM_PMM_LIST_ZERO 1
#define UVM_PMM_ALLOC_LIST_UNUSED 0
#define UVM_PMM_ALLOC_LIST_DISCARDED 1
#define UVM_PMM_ALLOC_LIST_USED 2
#define UVM_PMM_ALLOC_LIST_COUNT 3
#define U64_MAX UINT64_MAX
#define UVM_ASSERT assert
#define max(a,b) ((a)>(b)?(a):(b))
#define NV_GETTIME() mock_now
static NvU64 mock_now=1000;
static unsigned uvm_perf_eager_eviction_heat_ratio=2;
static unsigned uvm_perf_eager_eviction_stale_intervals=5;
static bool uvm_perf_migration_aware_eviction_enable=true;
struct chunk;
struct list_head { struct chunk *first; };
typedef struct chunk { struct { struct chunk *next; } list; bool is_zero, claimed; } uvm_gpu_chunk_t;
typedef struct { uvm_gpu_chunk_t chunk; NvU64 access_heat,last_access_epoch,protected_until_ns,resident_since_ns; unsigned migration_score; } uvm_gpu_root_chunk_t;
typedef struct { int lock,list_lock; struct { struct list_head alloc_list[3]; } root_chunks; struct { NvU64 epoch; } eager_eviction; struct list_head free_list[2]; } uvm_pmm_gpu_t;
static unsigned visits, evictions, freed, mock_usage, fail_at;
static int migration_status;
#define list_for_each_entry(entry,head,member) for((entry)=(head)->first;(entry) && (++visits,1);(entry)=(entry)->member.next)
static bool list_empty(struct list_head *h) {return !h->first;}
static uvm_gpu_chunk_t *list_first_chunk(struct list_head *h) {return h->first;}
static struct list_head *find_free_list(uvm_pmm_gpu_t *p,int type,int size,int zero) {assert(type==0 && size==UVM_CHUNK_SIZE_MAX);return &p->free_list[zero];}
static uvm_gpu_root_chunk_t *root_chunk_from_chunk(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *c) {(void)p;return (uvm_gpu_root_chunk_t *)c;}
static void uvm_assert_spinlock_locked(int *p) {assert(*p);}
static void uvm_spin_lock(int *p) {assert(!*p);*p=1;}
static void uvm_spin_unlock(int *p) {assert(*p);*p=0;}
static void uvm_mutex_lock(int *p) {assert(!*p);*p=1;}
static void uvm_mutex_unlock(int *p) {assert(*p);*p=0;}
static void remove_chunk(struct list_head *h,uvm_gpu_chunk_t *c) {uvm_gpu_chunk_t **p=&h->first;while(*p && *p!=c)p=&(*p)->list.next;if(*p)*p=c->list.next;}
static void chunk_start_eviction(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *c) {
    assert(p->lock && p->list_lock && !c->claimed);c->claimed=true;
    for(unsigned i=0;i<3;++i)remove_chunk(&p->root_chunks.alloc_list[i],c);
    for(unsigned i=0;i<2;++i)remove_chunk(&p->free_list[i],c);
}
static NV_STATUS evict_root_chunk(uvm_pmm_gpu_t *p,uvm_gpu_root_chunk_t *r,int context) {
    assert(p->lock && !p->list_lock && r->chunk.claimed && context==PMM_CONTEXT_DEFAULT);++evictions;return fail_at && evictions==fail_at ? 1 : migration_status;
}
static unsigned eager_eviction_usage_percent(uvm_pmm_gpu_t *p) {(void)p;return mock_usage;}
static unsigned eager_eviction_usage_percent_pending(uvm_pmm_gpu_t *p,unsigned pending) {(void)p;return mock_usage-pending;}
static void free_root_chunk(uvm_pmm_gpu_t *p,uvm_gpu_root_chunk_t *r,int mode) {assert(!p->lock && !p->list_lock && r->chunk.claimed && mode==0);++freed;--mock_usage;}
'''

SCENARIOS = r'''
static uvm_pmm_gpu_t p;
static uvm_gpu_root_chunk_t roots[6];
static void reset(void) {
    memset(&p,0,sizeof(p));memset(roots,0,sizeof(roots));p.eager_eviction.epoch=20;
    visits=evictions=freed=fail_at=0;mock_usage=100;migration_status=0;
    uvm_perf_migration_aware_eviction_enable=true;
}
static void add(unsigned i,unsigned list) {
    roots[i].chunk.list.next=p.root_chunks.alloc_list[list].first;
    p.root_chunks.alloc_list[list].first=&roots[i].chunk;
}
static bool admits(void) {
    uvm_spin_lock(&p.list_lock);
    bool result=eager_eviction_has_cold_candidate(&p);
    uvm_spin_unlock(&p.list_lock);
    return result;
}
int main(void) {
    uvm_gpu_root_chunk_t *out=NULL;
    reset();assert(!eager_eviction_one(&p,false,&out));assert(!eager_eviction_one(&p,true,&out));
    assert(!out && !p.lock && !p.list_lock);

    // Below high water, uniform hot data is not proactively evicted.
    // Recovery still reclaims it when forced, including migration grace.
    reset();add(0,2);roots[0].access_heat=8;roots[0].protected_until_ns=mock_now+99;
    assert(!admits() && !eager_eviction_one(&p,false,&out));
    assert(eager_eviction_one(&p,true,&out) && out==&roots[0]);

    // Zero heat needs the old stale-age threshold unless there is contrast.
    reset();add(0,2);roots[0].last_access_epoch=16;assert(!admits());
    roots[0].last_access_epoch=15;assert(admits());
    assert(eager_eviction_one(&p,false,&out) && out==&roots[0]);
    reset();add(0,2);add(1,2);roots[0].access_heat=4;roots[1].access_heat=7;
    assert(!admits());roots[1].access_heat=8;assert(admits());
    assert(eager_eviction_one(&p,false,&out) && out==&roots[0]);

    // Preserve the source branch's separate admission and migration ranking.
    // It can admit on a cold root and choose a hotter lower-score root.
    reset();add(0,2);add(1,2);roots[0].access_heat=64;roots[1].migration_score=4;
    assert(admits());assert(eager_eviction_one(&p,false,&out) && out==&roots[0]);

    // Free non-zero, free zero, unused, discarded, then used priority.
    reset();add(0,2);add(1,1);add(2,0);
    p.free_list[1].first=&roots[3].chunk;roots[3].chunk.is_zero=true;
    p.free_list[0].first=&roots[4].chunk;
    for(int i=4;i>=0;--i) assert(eager_eviction_one(&p,true,&out) && out==&roots[i]);
    assert(!eager_eviction_one(&p,true,&out));

    // Unprotected roots rank ahead of protected roots, even when hotter.
    reset();add(0,2);add(1,2);roots[0].protected_until_ns=mock_now+1;roots[1].access_heat=32;
    assert(eager_eviction_one(&p,true,&out) && out==&roots[1]);

    // Low-water target and successful-submission draining after a failure.
    reset();for(unsigned i=0;i<5;++i)add(i,2);
    assert(eager_eviction_batch(&p,true,16,98)==2 && evictions==2 && freed==2 && mock_usage==98);
    reset();for(unsigned i=0;i<5;++i)add(i,2);fail_at=2;
    assert(eager_eviction_batch(&p,true,16,95)==1 && evictions==2 && freed==1);
    assert(!p.lock && !p.list_lock);
    reset();add(0,2);roots[0].access_heat=8;
    assert(!eager_eviction_batch(&p,false,16,95) && !evictions && !freed);
    reset();add(0,2);migration_status=1;out=NULL;
    assert(!eager_eviction_one(&p,true,&out) && !out && !p.lock && !p.list_lock);
    puts("Demand-first production eviction admission, ranking, forced recovery and batch checks passed.");
    return 0;
}
'''

def main():
    source=(ROOT / 'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text()
    code=PRELUDE
    for name in ('get_first_allocated_chunk','pick_root_chunk_to_evict','eager_eviction_has_cold_candidate','eager_eviction_one','eager_eviction_batch'):
        code+=function(source,name)
    code+=SCENARIOS
    with tempfile.TemporaryDirectory(prefix='demand-first-eviction-') as directory:
        c=Path(directory)/'test.c';binary=Path(directory)/'test'
        c.write_text(code)
        subprocess.run(['cc','-std=gnu11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-g',str(c),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)

if __name__ == '__main__':
    main()
