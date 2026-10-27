#!/usr/bin/env python3
"""Exercise actual background/demand selectors with mocked lists and migration."""
from pathlib import Path
import subprocess
import tempfile
import re

def function(source, name):
    start = re.search(r"^static [^\n]*\b" + name + r"\(", source, re.M).start()
    end = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'kernel-open/nvidia-uvm/uvm_pmm_gpu.c'
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
static unsigned uvm_perf_eager_eviction_cold_threshold;
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
static void reset(void) {memset(&p,0,sizeof(p));memset(roots,0,sizeof(roots));p.eager_eviction.epoch=20;visits=evictions=freed=fail_at=0;mock_usage=100;migration_status=0;uvm_perf_eager_eviction_cold_threshold=0;uvm_perf_migration_aware_eviction_enable=true;}
static void add(unsigned i,unsigned list) {roots[i].chunk.list.next=p.root_chunks.alloc_list[list].first;p.root_chunks.alloc_list[list].first=&roots[i].chunk;}
static uvm_gpu_chunk_t *select_bg(bool recovering) {uvm_spin_lock(&p.list_lock);uvm_gpu_chunk_t *c=eager_eviction_pick_chunk_locked(&p,recovering);uvm_spin_unlock(&p.list_lock);return c;}
int main(void) {
    // Reproduce parent mismatch: cold existence admits a different hot victim.
    reset();roots[0].access_heat=64;roots[1].migration_score=4;add(0,2);add(1,2);
    uvm_spin_lock(&p.list_lock);
    assert(baseline_has_cold_candidate(&p));
    assert(get_first_allocated_chunk(&p)==&roots[0].chunk);
    uvm_spin_unlock(&p.list_lock);
    assert(visits==4);
    visits=0;assert(select_bg(false)==&roots[1].chunk && visits==2);
    uvm_gpu_root_chunk_t *out=NULL;
    assert(eager_eviction_one(&p,true,&out) && out==&roots[1] && evictions==1);
    assert(!p.lock && !p.list_lock && roots[1].chunk.claimed);
    assert(!eager_eviction_one(&p,true,&out) && evictions==1);

    // All-hot: both background modes refuse; demand still claims the hot root.
    reset();roots[0].access_heat=1;add(0,2);
    assert(!select_bg(false) && !select_bg(true));
    uvm_mutex_lock(&p.lock);assert(pick_root_chunk_to_evict(&p)==&roots[0]);uvm_mutex_unlock(&p.lock);
    assert(roots[0].chunk.claimed);

    // Grace protection applies even to zero heat during recovery; exact expiry.
    reset();roots[0].protected_until_ns=mock_now+1;add(0,2);assert(!select_bg(true));
    roots[0].protected_until_ns=mock_now;assert(select_bg(true)==&roots[0].chunk);
    roots[0].protected_until_ns=mock_now+1;uvm_perf_migration_aware_eviction_enable=false;assert(select_bg(true)==&roots[0].chunk);

    // Full-width heat, inclusive cutoff, ratio boundary, zero and stale cases.
    reset();add(0,2);roots[0].access_heat=UINT64_MAX;uvm_perf_eager_eviction_cold_threshold=UINT_MAX;
    assert(!select_bg(true));roots[0].access_heat=UINT_MAX;assert(select_bg(true));
    roots[0].access_heat=4;uvm_perf_eager_eviction_cold_threshold=3;assert(!select_bg(true));
    uvm_perf_eager_eviction_cold_threshold=4;assert(select_bg(true));assert(!select_bg(false));
    roots[1].access_heat=7;add(1,2);assert(!select_bg(false));roots[1].access_heat=8;assert(select_bg(false)==&roots[0].chunk);
    reset();add(0,2);roots[0].last_access_epoch=16;assert(!select_bg(false));assert(select_bg(true));
    roots[0].last_access_epoch=15;assert(select_bg(false));

    // A fresh zero root must not conceal a stale zero root with higher score.
    reset();add(0,2);add(1,2);roots[0].last_access_epoch=20;roots[1].last_access_epoch=0;roots[1].migration_score=4;
    assert(select_bg(false)==&roots[1].chunk);assert(select_bg(true)==&roots[0].chunk);

    // Colder roots first, then migration score/residency/last-access tie-breaks.
    reset();uvm_perf_eager_eviction_cold_threshold=4;add(0,2);add(1,2);
    roots[0].access_heat=1;roots[1].migration_score=9;assert(select_bg(true)==&roots[1].chunk);
    roots[0].access_heat=0;assert(select_bg(true)==&roots[0].chunk);
    roots[1].migration_score=0;roots[0].resident_since_ns=5;assert(select_bg(true)==&roots[1].chunk);
    roots[0].resident_since_ns=0;roots[1].last_access_epoch=1;assert(select_bg(true)==&roots[0].chunk);

    // Free, unused and discarded roots bypass used-data heat/grace protection.
    reset();for(unsigned i=0;i<5;++i){roots[i].access_heat=UINT64_MAX;roots[i].protected_until_ns=mock_now+99;}
    add(0,2);add(1,1);add(2,0);p.free_list[1].first=&roots[3].chunk;roots[3].chunk.is_zero=true;p.free_list[0].first=&roots[4].chunk;
    for(int i=4;i>=1;--i){out=NULL;assert(eager_eviction_one(&p,true,&out) && out==&roots[i]);}
    assert(!eager_eviction_one(&p,true,&out));

    // Draining a short batch remains mandatory when only hot roots remain.
    reset();for(unsigned i=0;i<5;++i)add(i,2);roots[0].access_heat=64;
    assert(eager_eviction_batch(&p,true,16,95)==4 && evictions==4 && freed==4 && mock_usage==96);
    assert(!roots[0].chunk.claimed);
    reset();for(unsigned i=0;i<5;++i)add(i,2);
    assert(eager_eviction_batch(&p,true,16,98)==2 && evictions==2 && freed==2);
    reset();for(unsigned i=0;i<5;++i)add(i,2);fail_at=2;
    assert(eager_eviction_batch(&p,true,16,95)==1 && evictions==2 && freed==1);

    // Empty sets and failed migration do not publish a successful victim.
    reset();assert(!eager_eviction_one(&p,true,&out));add(0,2);migration_status=1;out=NULL;
    assert(!eager_eviction_one(&p,true,&out) && !out && !p.lock && !p.list_lock);
    puts("Passed: parent mismatch reproduced; cold cutoff/grace, scan/claim, free priority, emergency fallback and error paths.");
    return 0;
}
'''


def main():
    source = (ROOT / SOURCE).read_text()
    original = subprocess.check_output(['git','show','12ceb155:'+SOURCE], cwd=ROOT, text=True)
    code = PRELUDE
    for name in ('get_first_allocated_chunk','pick_root_chunk_to_evict','eager_eviction_root_is_colder','eager_eviction_pick_chunk_locked','eager_eviction_one','eager_eviction_batch'):
        code += function(source,name)
    code += function(original,'eager_eviction_has_cold_candidate').replace('eager_eviction_has_cold_candidate(', 'baseline_has_cold_candidate(')
    code += SCENARIOS
    with tempfile.TemporaryDirectory(prefix='cold-only-eviction-') as directory:
        path=Path(directory)
        (path/'test.c').write_text(code)
        subprocess.run(['cc','-std=gnu11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-g',str(path/'test.c'),'-o',str(path/'test')],check=True)
        subprocess.run([str(path/'test')],check=True)


if __name__ == '__main__':
    main()
