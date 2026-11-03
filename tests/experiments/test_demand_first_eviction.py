#!/usr/bin/env python3
"""Execute the demand-first branch's production eviction functions with mocked PMM dependencies."""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).parents[2]

def function(source, name):
    match = re.search(r"^(?:static [^\n]*|void )\b" + name + r"\(", source, re.M)
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
typedef unsigned NvU32;
/* VAB classifier */
typedef unsigned uvm_pmm_alloc_list_t;
typedef int NV_STATUS;
#define NV_OK 0
#define ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))
#define FREE_ROOT_CHUNK_MODE_DEFAULT 0
#define PMM_CONTEXT_DEFAULT 0
#define UVM_PMM_GPU_MEMORY_TYPE_USER 0
#define UVM_PMM_GPU_CHUNK_STATE_ALLOCATED 1
#define UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT 2
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
typedef struct chunk { struct { struct chunk *next; } list; bool is_zero, claimed; NvU64 address; unsigned state; void *va_block; } uvm_gpu_chunk_t;
typedef struct { uvm_gpu_chunk_t chunk; NvU64 access_heat,last_access_epoch,protected_until_ns,resident_since_ns; unsigned migration_score; NvU64 vab_generation,vab_ready_generation,vab_ready_ns; } uvm_gpu_root_chunk_t;
typedef struct { NvU64 used_cold,used_hot,used_unknown,selected_hot,selected_cold,selected_unknown,background_roots_freed,background_failures,usage_percent,prefetch_inflight,recovering_watermark,background_active,selection_changes,selection_fallbacks,eligible_roots,readiness_pending; } uvm_pmm_gpu_vab_stats_t;
typedef struct { int dummy; } uvm_parent_gpu_t;
typedef struct { uvm_parent_gpu_t *parent; } uvm_gpu_t;
static uvm_gpu_t mock_gpu;
typedef struct { int lock,list_lock; struct { struct list_head alloc_list[3]; } root_chunks; struct { NvU64 epoch; unsigned prefetch_count; bool recovering_watermark, active; } eager_eviction; struct list_head free_list[2]; uvm_pmm_gpu_vab_stats_t vab_stats; } uvm_pmm_gpu_t;
static uvm_gpu_t *uvm_pmm_to_gpu(uvm_pmm_gpu_t *p) {(void)p; return &mock_gpu;}
static uvm_vab_snapshot_t mock_snapshot;
static unsigned snapshot_calls;
static void uvm_gpu_vab_get_snapshot(uvm_parent_gpu_t *p,uvm_vab_snapshot_t *s) {
    (void)p; ++snapshot_calls; *s=mock_snapshot;
    s->valid=s->valid && uvm_vab_sample_fresh(mock_now,s->sampled_ns,300);
}
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
    memset(&mock_snapshot,0,sizeof(mock_snapshot));snapshot_calls=0;
    for(unsigned i=0;i<6;++i){roots[i].chunk.address=(NvU64)i*UVM_CHUNK_SIZE_MAX;
        roots[i].chunk.state=UVM_PMM_GPU_CHUNK_STATE_ALLOCATED;roots[i].chunk.va_block=(void *)1;}
    mock_now=1000;
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
static void observation(void) {
    mock_snapshot.valid=true;
    mock_snapshot.sampled_ns=mock_now;mock_snapshot.linear_base=0;mock_snapshot.granularity=5;
    for(unsigned i=0;i<64;++i)mock_snapshot.enabled_mask[i]=~0ULL;
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

    // Observe mode reports the chosen root but never changes the old ranking.
    reset();add(0,2);add(1,2);roots[1].migration_score=9;observation();
    mock_snapshot.access_bits[0]=1;
    assert(eager_eviction_one(&p,true,&out) && out==&roots[0]);
    assert(snapshot_calls==1 && p.vab_stats.selected_hot==1);

    // Cold/hot/masked observations never change the migration-aware ranking.
    reset();add(0,2);add(1,2);roots[1].migration_score=9;observation();
    mock_snapshot.access_bits[0]=2;
    assert(eager_eviction_one(&p,true,&out) && out==&roots[0]);
    assert(p.vab_stats.selected_cold==1);
    reset();add(0,2);add(1,2);roots[1].migration_score=9;observation();
    mock_snapshot.access_bits[0]=1;mock_snapshot.enabled_mask[0]&=~1ULL;
    assert(eager_eviction_one(&p,true,&out) && out==&roots[0]);
    assert(p.vab_stats.selected_unknown==1);
    reset();add(0,2);add(1,2);roots[1].migration_score=9;observation();
    mock_snapshot.access_bits[0]=3;
    assert(eager_eviction_one(&p,true,&out) && out==&roots[0]);
    assert(p.vab_stats.selected_hot==1);

    // Newly resident data is unknown when a known timestamp follows sampling.
    reset();add(0,2);add(1,2);roots[1].migration_score=9;observation();
    roots[0].resident_since_ns=mock_now+1;
    assert(eager_eviction_one(&p,true,&out) && out==&roots[0]);
    assert(p.vab_stats.selected_unknown==1);

    // Observing cold bits does not admit any extra background evictions.
    reset();add(0,2);add(1,2);roots[0].access_heat=roots[1].access_heat=4;
    observation();mock_snapshot.access_bits[0]=1;assert(!admits());
    mock_snapshot.access_bits[0]=0;assert(!admits());

    // The procfs view counts invalid/unmapped observations as unknown.
    uvm_pmm_gpu_vab_stats_t stats;
    mock_snapshot.valid=false;uvm_pmm_gpu_vab_stats(&p,&mock_snapshot,&stats);
    assert(stats.used_unknown==2 && !stats.used_hot && !stats.used_cold);
    observation();mock_snapshot.access_bits[0]=1;
    uvm_pmm_gpu_vab_stats(&p,&mock_snapshot,&stats);
    assert(stats.used_hot==1 && stats.used_cold==1 && !stats.used_unknown);

    reset();for(unsigned i=0;i<5;++i)add(i,2);
    assert(eager_eviction_batch(&p,true,16,98)==2 && p.vab_stats.background_roots_freed==2);
    reset();for(unsigned i=0;i<5;++i)add(i,2);fail_at=2;
    assert(eager_eviction_batch(&p,true,16,95)==1);
    assert(p.vab_stats.background_roots_freed==1 && p.vab_stats.background_failures==1);

    // Mode 2 changes the actual victim, while retaining the baseline winner
    // for exact counterfactual change accounting.
    reset();add(0,2);add(1,2);roots[1].migration_score=9;observation();mock_snapshot.eviction_enabled=true;
    mock_snapshot.access_bits[0]=1;
    for(unsigned i=0;i<2;++i){roots[i].vab_generation=roots[i].vab_ready_generation=1;roots[i].vab_ready_ns=900;}
    assert(eager_eviction_one(&p,true,&out)&&out==&roots[1]);
    assert(p.vab_stats.selection_changes==1&&!p.vab_stats.selection_fallbacks&&p.vab_stats.selected_cold==1);

    // No 'changed decision' is claimed when baseline also selects cold data.
    reset();add(0,2);add(1,2);roots[1].migration_score=9;observation();mock_snapshot.eviction_enabled=true;
    mock_snapshot.access_bits[0]=2;
    for(unsigned i=0;i<2;++i){roots[i].vab_generation=roots[i].vab_ready_generation=1;roots[i].vab_ready_ns=900;}
    assert(eager_eviction_one(&p,true,&out)&&out==&roots[0]);
    assert(!p.vab_stats.selection_changes&&!p.vab_stats.selection_fallbacks);

    // All-hot pressure recovery falls back, and fresh protected cold data
    // does not displace an unprotected legacy candidate.
    reset();add(0,2);add(1,2);roots[1].migration_score=9;observation();mock_snapshot.eviction_enabled=true;
    mock_snapshot.access_bits[0]=3;
    for(unsigned i=0;i<2;++i){roots[i].vab_generation=roots[i].vab_ready_generation=1;roots[i].vab_ready_ns=900;}
    assert(eager_eviction_one(&p,true,&out)&&out==&roots[0]&&p.vab_stats.selection_fallbacks==1);
    reset();add(0,2);add(1,2);observation();mock_snapshot.eviction_enabled=true;mock_snapshot.access_bits[0]=1;
    roots[1].protected_until_ns=mock_now+100;
    for(unsigned i=0;i<2;++i){roots[i].vab_generation=roots[i].vab_ready_generation=1;roots[i].vab_ready_ns=900;}
    assert(eager_eviction_one(&p,true,&out)&&out==&roots[0]);

    // Background admission below high water claims the exact VAB-cold root,
    // even though its migration score would lose the legacy ranking.
    reset();add(0,2);add(1,2);roots[0].access_heat=roots[1].access_heat=4;roots[1].migration_score=9;
    observation();mock_snapshot.eviction_enabled=true;mock_snapshot.access_bits[0]=1;
    for(unsigned i=0;i<2;++i){roots[i].vab_generation=roots[i].vab_ready_generation=1;roots[i].vab_ready_ns=900;}
    assert(!admits());assert(eager_eviction_one(&p,false,&out)&&out==&roots[1]);
    reset();add(0,2);add(1,2);observation();mock_snapshot.eviction_enabled=true;mock_snapshot.access_bits[0]=3;
    for(unsigned i=0;i<2;++i){roots[i].vab_generation=roots[i].vab_ready_generation=1;roots[i].vab_ready_ns=900;}
    assert(!eager_eviction_one(&p,false,&out)&&!evictions);

    // Arrivals invalidate readiness independently of old migration scoring.
    reset();add(0,2);add(1,2);roots[1].migration_score=9;observation();mock_snapshot.eviction_enabled=true;
    mock_snapshot.access_bits[0]=1;uvm_perf_migration_aware_eviction_enable=false;
    for(unsigned i=0;i<2;++i){roots[i].vab_generation=roots[i].vab_ready_generation=1;roots[i].vab_ready_ns=900;}
    vab_root_changed(&roots[1]);assert(roots[1].vab_generation==2&&!roots[1].vab_ready_ns);
    uvm_spin_lock(&p.list_lock);assert(vab_root_state(&roots[1],&mock_snapshot)==UVM_VAB_REGION_UNKNOWN);uvm_spin_unlock(&p.list_lock);
    roots[1].vab_ready_generation=2;roots[1].vab_ready_ns=1010;
    assert(vab_root_state(&roots[1],&mock_snapshot)==UVM_VAB_REGION_UNKNOWN);
    mock_snapshot.sampled_ns=1020;mock_now=1030;
    assert(vab_root_state(&roots[1],&mock_snapshot)==UVM_VAB_REGION_COLD);
    assert(eager_eviction_one(&p,true,&out)&&out==&roots[1]);

    // Split, unmapped, stale and unready roots never become VAB-cold.
    reset();observation();mock_snapshot.eviction_enabled=true;
    roots[0].vab_generation=roots[0].vab_ready_generation=1;roots[0].vab_ready_ns=900;
    roots[0].chunk.state=UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT;
    assert(vab_root_state(&roots[0],&mock_snapshot)==UVM_VAB_REGION_UNKNOWN);
    roots[0].chunk.state=UVM_PMM_GPU_CHUNK_STATE_ALLOCATED;roots[0].chunk.va_block=NULL;
    assert(vab_root_state(&roots[0],&mock_snapshot)==UVM_VAB_REGION_UNKNOWN);

    // Exercise the unchanged allocation/PMA wrapper and its fallback path.
    reset();add(0,2);uvm_mutex_lock(&p.lock);assert(pick_root_chunk_to_evict(&p)==&roots[0]);uvm_mutex_unlock(&p.lock);
    puts("Demand-first production eviction admission, ranking, forced recovery and batch checks passed.");
    return 0;
}
'''

def main():
    source=(ROOT / 'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text()
    header=(ROOT / 'kernel-open/nvidia-uvm/uvm_vab.h').read_text().replace('#include "nvtypes.h"','')
    code=PRELUDE.replace('/* VAB classifier */',header)
    for name in ('vab_root_changed','vab_candidate_better','vab_root_state','vab_pick_allocated','get_first_allocated_chunk','vab_pick_root','pick_root_chunk_to_evict','eager_eviction_has_cold_candidate','eager_eviction_one','eager_eviction_batch','uvm_pmm_gpu_vab_stats'):
        code+=function(source,name)
    code+=SCENARIOS
    with tempfile.TemporaryDirectory(prefix='demand-first-eviction-') as directory:
        c=Path(directory)/'test.c';binary=Path(directory)/'test'
        c.write_text(code)
        subprocess.run(['cc','-std=gnu11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-g',str(c),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)

if __name__ == '__main__':
    main()
