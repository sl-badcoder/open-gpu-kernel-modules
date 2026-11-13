#!/usr/bin/env python3
"""Execute mandatory VAB-ranked reclaim and passive diagnostics."""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).parents[2]

def function(source, name):
    match = re.search(r"^(?:static [^\n]*|void |NvU64 |bool |NV_STATUS )\b" + name + r"\(", source, re.M)
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
#define UVM_ASSERT assert
#define UVM_SIZE_1MB (1024ULL * 1024ULL)
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
#define min(a,b) ((a)<(b)?(a):(b))
#define NV_GETTIME() mock_now
static NvU64 mock_now=1000;
#define NSEC_PER_MSEC 1000000ULL
struct chunk;
struct list_head { struct chunk *first; };
typedef struct chunk { struct { struct chunk *next; } list; bool is_zero, claimed; NvU64 address; unsigned state; void *va_block; } uvm_gpu_chunk_t;
typedef struct { uvm_gpu_chunk_t chunk; NvU64 vab_generation,vab_ready_generation,vab_ready_ns; } uvm_gpu_root_chunk_t;
/* VAB stats */
typedef struct { bool access_counters_supported, is_integrated_gpu; } uvm_parent_gpu_t;
static uvm_parent_gpu_t mock_parent;
typedef struct { uvm_parent_gpu_t *parent; } uvm_gpu_t;
static uvm_gpu_t mock_gpu;
typedef struct { NvU64 numPages2m,numFreePages64k; } pma_stats_t;
#define UVM_CHUNK_SIZE_2M UVM_CHUNK_SIZE_MAX
#define UVM_PAGE_SIZE_64K 65536ULL
#define READ_ONCE(x) (x)
typedef struct { int lock,list_lock; pma_stats_t *pma_stats; struct { struct list_head alloc_list[3]; } root_chunks; struct list_head free_list[2]; uvm_pmm_gpu_vab_stats_t vab_stats; } uvm_pmm_gpu_t;
static uvm_gpu_t *uvm_pmm_to_gpu(uvm_pmm_gpu_t *p) {(void)p; return &mock_gpu;}
static uvm_vab_snapshot_t mock_snapshot;
static unsigned snapshot_calls;
static void uvm_gpu_vab_get_snapshot(uvm_parent_gpu_t *p,uvm_vab_snapshot_t *s) {
    (void)p; ++snapshot_calls; *s=mock_snapshot;
    s->valid=s->valid && uvm_vab_sample_fresh(mock_now,s->sampled_ns,300);
}
static NvU64 mock_recency[6];
static bool mock_recency_race;
static bool uvm_gpu_vab_get_recency(uvm_parent_gpu_t *p,const uvm_vab_snapshot_t *s,
                                    NvU64 address,NvU64 size,NvU64 ready,NvU64 *last) {
    (void)p; (void)size;
    if(mock_recency_race || uvm_vab_classify(s,address,size,ready)==UVM_VAB_REGION_UNKNOWN)return false;
    *last=max(ready,mock_recency[address/UVM_CHUNK_SIZE_MAX]); return true;
}
static unsigned visits;
#define list_for_each_entry(entry,head,member) for((entry)=(head)->first;(entry) && (++visits,1);(entry)=(entry)->member.next)
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
'''

SCENARIOS = r'''
static uvm_pmm_gpu_t p;
static pma_stats_t memory;
static uvm_gpu_root_chunk_t roots[6];
static void reset(void)
{
 memset(&p,0,sizeof(p));memset(roots,0,sizeof(roots));memset(&mock_snapshot,0,sizeof(mock_snapshot));
 mock_parent=(uvm_parent_gpu_t){true,false};mock_gpu.parent=&mock_parent;
 memory=(pma_stats_t){8192,131072};p.pma_stats=&memory;
 mock_now=1000000000ULL;visits=snapshot_calls=0;
 memset(mock_recency,0,sizeof(mock_recency));mock_recency_race=false;
 mock_snapshot.valid=mock_snapshot.eviction_enabled=mock_snapshot.lru_enabled=true;
 mock_snapshot.sampled_ns=mock_now;mock_snapshot.linear_base=0;mock_snapshot.granularity=5;
 for(unsigned i=0;i<64;++i)mock_snapshot.enabled_mask[i]=~0ULL;
 for(unsigned i=0;i<6;++i) {
  roots[i].chunk.address=(NvU64)i*UVM_CHUNK_SIZE_MAX;
  roots[i].chunk.state=UVM_PMM_GPU_CHUNK_STATE_ALLOCATED;roots[i].chunk.va_block=(void *)1;
  roots[i].vab_generation=roots[i].vab_ready_generation=1;roots[i].vab_ready_ns=mock_now-10*NSEC_PER_MSEC;
 }
}
static void add(unsigned i,unsigned list)
{ roots[i].chunk.list.next=p.root_chunks.alloc_list[list].first;p.root_chunks.alloc_list[list].first=&roots[i].chunk; }
static uvm_gpu_root_chunk_t *pick(void)
{ uvm_gpu_root_chunk_t *r;uvm_mutex_lock(&p.lock);r=pick_root_chunk_to_evict(&p);uvm_mutex_unlock(&p.lock);return r; }
int main(void)
{
 // No fixed background grace: usable young roots can be ranked on demand.
 reset();add(0,2);add(1,2);mock_snapshot.access_bits[0]=3;
 roots[1].vab_ready_ns=mock_now-5*NSEC_PER_MSEC;
 mock_recency[0]=mock_now-8*NSEC_PER_MSEC;mock_recency[1]=mock_now-2*NSEC_PER_MSEC;
 assert(pick()==&roots[0] && p.vab_stats.selected_hot==1 && p.vab_stats.selection_changes==1);
 assert(!p.vab_stats.selection_fallbacks && roots[0].chunk.claimed);
 // Current-sample HOT ties lose to a known COLD candidate.
 reset();add(0,2);add(1,2);mock_snapshot.access_bits[0]=2;
 mock_recency[0]=mock_recency[1]=mock_now;
 assert(pick()==&roots[0] && p.vab_stats.selected_cold==1);
 // Cold-first mode remains supported by VAB; mode 3 uses recency for all known roots.
 reset();add(0,2);add(1,2);mock_snapshot.lru_enabled=false;mock_snapshot.access_bits[0]=2;
 assert(pick()==&roots[0]);
 // Unmapped, stale, split, recycled, ownerless, and racing evidence cannot rank
 // roots. Mandatory reclaim falls back to list order for forward progress.
 for(unsigned reason=0;reason<7;++reason) {
  reset();add(0,2);
  if(reason==0)mock_snapshot.linear_base=~0ULL;
  if(reason==1)mock_snapshot.sampled_ns=mock_now-301;
  if(reason==2)roots[0].chunk.state=UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT;
  if(reason==3)roots[0].chunk.va_block=NULL;
  if(reason==4)vab_root_changed(&roots[0]);
  if(reason==5)mock_recency_race=true;
  if(reason==6)mock_snapshot.valid=false;
  assert(pick()==&roots[0] && p.vab_stats.selection_fallbacks==1);
 }
 // Observation-only mode uses the baseline order, without pretending VAB chose it.
 reset();add(0,2);add(1,2);mock_snapshot.eviction_enabled=false;
 assert(pick()==&roots[1] && !p.vab_stats.selection_changes && !p.vab_stats.selection_fallbacks);
 // Free, UNUSED and DISCARD lists still have priority over live used roots.
 reset();add(0,2);add(1,1);add(2,0);p.free_list[0].first=&roots[3].chunk;
 assert(pick()==&roots[3]);assert(pick()==&roots[2]);assert(pick()==&roots[1]);assert(pick()==&roots[0]);
 reset();assert(!pick());
 // Passive diagnostics preserve all state and report the actual memory use.
 reset();add(0,2);add(1,2);mock_snapshot.access_bits[0]=2;
 uvm_gpu_root_chunk_t before[6];memcpy(before,roots,sizeof(roots));
 uvm_pmm_gpu_vab_stats_t stats;uvm_pmm_gpu_vab_stats(&p,&mock_snapshot,&stats);
 assert(stats.used_cold==1 && stats.used_hot==1 && !stats.used_unknown && stats.eligible_roots==1);
 assert(stats.usage_percent==50 && !memcmp(before,roots,sizeof(roots)));
 memory.numFreePages64k=~0ULL;assert(pmm_gpu_usage_percent(&p)==0);
 memory.numPages2m=0;assert(pmm_gpu_usage_percent(&p)==0);
 p.pma_stats=NULL;assert(pmm_gpu_usage_percent(&p)==0);
 puts("Mandatory VAB ranking: young roots, cold ties, fallback, free-list priority and passive telemetry passed.");
 return 0;
}
'''

def main():
    source=(ROOT / 'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text()
    header=(ROOT / 'kernel-open/nvidia-uvm/uvm_vab.h').read_text().replace('#include "nvtypes.h"','')
    code=PRELUDE.replace('/* VAB classifier */',header)
    pmm_header = (ROOT / 'kernel-open/nvidia-uvm/uvm_pmm_gpu.h').read_text()
    stats_end = pmm_header.index('} uvm_pmm_gpu_vab_stats_t;') + len('} uvm_pmm_gpu_vab_stats_t;')
    stats = pmm_header[pmm_header.rfind('typedef struct', 0, stats_end):stats_end]
    code = code.replace('/* VAB stats */', stats)
    for name in ('vab_root_changed','vab_root_state','vab_pick_allocated','get_first_allocated_chunk',
                 'pick_root_chunk_to_evict','pmm_gpu_usage_percent','uvm_pmm_gpu_vab_stats'):
        code+=function(source,name)
    code+=SCENARIOS
    with tempfile.TemporaryDirectory(prefix='demand-first-eviction-') as directory:
        c=Path(directory)/'test.c';binary=Path(directory)/'test'
        c.write_text(code)
        subprocess.run(['cc','-std=gnu11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-g',str(c),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)

if __name__ == '__main__':
    main()
