#!/usr/bin/env python3
"""Execute the demand-first branch's production eviction functions with mocked PMM dependencies."""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).parents[2]

def function(source, name):
    matches = re.finditer(r"^(?:static [^\n]*|void |unsigned |NvU64 )\b" + name + r"\(", source, re.M)
    match = next((m for m in matches if ';' not in source[m.start():source.index('{', m.start())]), None)
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
typedef uint16_t NvU16;
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
#define UVM_PMM_GPU_CHUNK_STATE_FREE 0
#define READ_ONCE(x) (x)
#define UVM_PAGE_SIZE_64K 65536
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
typedef enum { UVM_PERF_DIAG_COUNTER_SERVICE, UVM_PERF_DIAG_POPULATION_SUBMIT,
               UVM_PERF_DIAG_POPULATION_WAIT, UVM_PERF_DIAG_EVICTION_SELECT,
               UVM_PERF_DIAG_EVICTION_SUBMIT, UVM_PERF_DIAG_EVICTION_DRAIN,
               UVM_PERF_DIAG_STAGE_COUNT } uvm_perf_diag_stage_t;
static unsigned diag_counts[UVM_PERF_DIAG_STAGE_COUNT];
static NvU64 uvm_perf_diag_begin(void) { return 1; }
static void uvm_perf_diag_end(uvm_perf_diag_stage_t stage, NvU64 start)
{ if (start) ++diag_counts[stage]; }
static bool uvm_perf_adaptive_replacement_admission_enable;
static unsigned uvm_perf_vab_min_residency_ms=100;
static unsigned uvm_perf_speculative_prefetch_limit_mb=32;
static unsigned uvm_perf_speculative_prefetch_reserve_mb=64;
static bool uvm_perf_prefetch_refault_enable=false;
static bool uvm_perf_prefetch_replacement_enable=false;
static bool uvm_pmm_gpu_refault_enabled(void) { return uvm_perf_prefetch_refault_enable; }
static unsigned uvm_perf_prefetch_refault_short_percent=25;
static unsigned uvm_perf_prefetch_refault_max_neighbors=1;
#define NSEC_PER_MSEC 1000000ULL
struct chunk;
struct list_head { struct chunk *first; };
typedef struct chunk { struct { struct chunk *next; } list; bool is_zero, claimed; NvU64 address; unsigned state; void *va_block; } uvm_gpu_chunk_t;
typedef struct { uvm_gpu_chunk_t chunk; NvU64 vab_generation,vab_ready_generation,vab_ready_ns; } uvm_gpu_root_chunk_t;
/* VAB stats */
typedef struct { int dummy; } uvm_parent_gpu_t;
typedef struct { uvm_parent_gpu_t *parent; struct { NvU64 size; struct {bool enabled;} numa; } mem_info; } uvm_gpu_t;
static uvm_gpu_t mock_gpu;
typedef struct { int lock,list_lock; struct { struct list_head alloc_list[3]; NvU64 cached_free_root_count; } root_chunks; struct mock_pma_stats {NvU64 numFreePages64k;} *pma_stats; struct { NvU64 epoch; unsigned replacement_roots, adaptive_level, adaptive_samples, adaptive_observed; NvU64 adaptive_feedback_samples, adaptive_feedback_observed, adaptive_candidates, adaptive_probes; NvU64 replacement_requests, replacement_freed_roots, cached_root_releases, allocated_root_releases; unsigned prefetch_count; bool recovering_watermark, active; uvm_prefetch_budget_t budget; uvm_prefetch_refault_t refault; struct {NvU64 events, confirmations, replacements, predictions, learned_predictions, expirations;} patterns; } eager_eviction; struct list_head free_list[2]; uvm_pmm_gpu_vab_stats_t vab_stats; } uvm_pmm_gpu_t;
static uvm_gpu_t *uvm_pmm_to_gpu(uvm_pmm_gpu_t *p) {(void)p; return &mock_gpu;}
static uvm_vab_snapshot_t mock_snapshot;
static unsigned snapshot_calls;
static void uvm_gpu_vab_get_snapshot(uvm_parent_gpu_t *p,uvm_vab_snapshot_t *s) {
    (void)p; ++snapshot_calls; *s=mock_snapshot;
    s->valid=s->valid && uvm_vab_sample_fresh(mock_now,s->sampled_ns,300);
}
static NvU64 mock_recency[256];
static bool mock_recency_race;
static bool uvm_gpu_vab_get_recency(uvm_parent_gpu_t *p,const uvm_vab_snapshot_t *s,
                                    NvU64 address,NvU64 size,NvU64 ready,NvU64 *last) {
    (void)p; (void)size;
    if(mock_recency_race || uvm_vab_classify(s,address,size,ready)==UVM_VAB_REGION_UNKNOWN)return false;
    *last=max(ready,mock_recency[address/UVM_CHUNK_SIZE_MAX]); return true;
}
static unsigned visits, evictions, freed, mock_usage, fail_at;
static int migration_status;
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
static bool chunk_is_evictable(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *c) {assert(p->list_lock);return !c->claimed && c->state==UVM_PMM_GPU_CHUNK_STATE_ALLOCATED;}
static NV_STATUS evict_root_chunk(uvm_pmm_gpu_t *p,uvm_gpu_root_chunk_t *r,int context) {
    assert(p->lock && !p->list_lock && r->chunk.claimed && context==PMM_CONTEXT_DEFAULT);++evictions;return fail_at && evictions==fail_at ? 1 : migration_status;
}
static unsigned eager_eviction_usage_percent(uvm_pmm_gpu_t *p) {(void)p;return mock_usage;}
static unsigned eager_eviction_usage_percent_pending(uvm_pmm_gpu_t *p,unsigned pending) {(void)p;return mock_usage-pending;}
static void free_root_chunk(uvm_pmm_gpu_t *p,uvm_gpu_root_chunk_t *r,int mode) {assert(!p->lock && !p->list_lock && r->chunk.claimed && mode==0);++freed;--mock_usage;}
'''

SCENARIOS = r'''
static bool one(uvm_pmm_gpu_t *p,uvm_gpu_root_chunk_t **out) {
 uvm_eviction_candidates_t candidates={.limit=1};
 return eager_eviction_one(p,out,&candidates);
}
static uvm_pmm_gpu_t p;
static uvm_gpu_root_chunk_t roots[256];
static void reset(void){
 memset(&p,0,sizeof(p));memset(roots,0,sizeof(roots));memset(&mock_snapshot,0,sizeof(mock_snapshot));
 mock_now=1000000000ULL;uvm_perf_vab_min_residency_ms=100;uvm_perf_prefetch_replacement_enable=false;
 visits=evictions=freed=fail_at=0;mock_usage=100;migration_status=0;snapshot_calls=0;
 memset(mock_recency,0,sizeof(mock_recency));mock_recency_race=false;
 mock_snapshot.valid=true;mock_snapshot.eviction_enabled=true;mock_snapshot.lru_enabled=true;
 mock_snapshot.sampled_ns=mock_now;mock_snapshot.linear_base=0;mock_snapshot.granularity=5;
 for(unsigned i=0;i<64;++i)mock_snapshot.enabled_mask[i]=~0ULL;
 for(unsigned i=0;i<256;++i){
  roots[i].chunk.address=(NvU64)i*UVM_CHUNK_SIZE_MAX;
  roots[i].chunk.state=UVM_PMM_GPU_CHUNK_STATE_ALLOCATED;roots[i].chunk.va_block=(void *)1;
  roots[i].vab_generation=roots[i].vab_ready_generation=1;roots[i].vab_ready_ns=mock_now-200*NSEC_PER_MSEC;
 }
}
static void add(unsigned i,unsigned list){
 roots[i].chunk.list.next=p.root_chunks.alloc_list[list].first;p.root_chunks.alloc_list[list].first=&roots[i].chunk;
}
int main(void){
 uvm_gpu_root_chunk_t *out=NULL;
 struct mock_pma_stats capacity={0};
 reset();
 p.pma_stats=&capacity;
 uvm_perf_prefetch_replacement_enable=true;
 uvm_spin_lock(&p.list_lock);
 for (unsigned i=0;i<16;++i)
     adaptive_replacement_feedback(&p,false);
 uvm_spin_unlock(&p.list_lock);
 assert(p.eager_eviction.adaptive_level==1 && p.eager_eviction.adaptive_feedback_samples==16);
 for (unsigned i=0;i<15;++i)
     assert(uvm_pmm_gpu_adaptive_admission_extra(&p,true)==2);
 assert(uvm_pmm_gpu_adaptive_admission_extra(&p,true)==0 && p.eager_eviction.adaptive_probes==1);
 capacity.numFreePages64k=4096;
 assert(uvm_pmm_gpu_adaptive_admission_extra(&p,true)==0 && p.eager_eviction.adaptive_candidates==16);
 uvm_spin_lock(&p.list_lock);
 for (unsigned i=0;i<16;++i)
     adaptive_replacement_feedback(&p,true);
 uvm_spin_unlock(&p.list_lock);
 assert(p.eager_eviction.adaptive_level==0 && p.eager_eviction.adaptive_feedback_observed==16);

 // Background skips all-hot roots.
 reset();add(0,2);add(1,2);mock_snapshot.access_bits[0]=3;
 assert(!one(&p,&out)&&!evictions);
 assert(p.vab_stats.background_hot_skips==2);
 // With one hot and one cold region, background selects the cold root.
 reset();add(0,2);add(1,2);mock_snapshot.access_bits[0]=(1ULL << 1);
 assert(one(&p,&out)&&out==&roots[0]);
 assert(p.vab_stats.selected_cold==1);
 // Cold-only behavior also applies in mode 2; observation-only never reclaims.
 reset();add(0,2);add(1,2);mock_snapshot.lru_enabled=false;mock_snapshot.access_bits[0]=2;
 assert(one(&p,&out)&&out==&roots[0]);
 reset();add(0,2);mock_snapshot.eviction_enabled=false;assert(!one(&p,&out));
 reset();add(0,2);mock_snapshot.valid=false;assert(!one(&p,&out));
 // Fixed completion grace protects roots below the residency threshold.
 reset();add(0,2);roots[0].vab_ready_ns=mock_now-99*NSEC_PER_MSEC;
 assert(!one(&p,&out)&&!evictions);
 // Unknown, stale, unmapped and racing data cannot admit proactive eviction.
 reset();add(0,2);vab_root_changed(&roots[0]);assert(!one(&p,&out));
 reset();add(0,2);mock_snapshot.sampled_ns=mock_now-301;assert(!one(&p,&out));
 reset();add(0,2);mock_snapshot.linear_base=~0ULL;assert(!one(&p,&out));
 reset();add(0,2);mock_recency_race=true;assert(!one(&p,&out));
 // A valid cold snapshot permits background selection.
 reset();add(0,2);assert(one(&p,&out)&&out==&roots[0]);
 // Background VAB selection operates on allocated roots only.
 // Split/ownerless roots remain unknown and pressure uses the fallback.
 reset();add(0,2);roots[0].chunk.state=UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT;
 assert(vab_root_state(&roots[0],&mock_snapshot)==UVM_VAB_REGION_UNKNOWN);
 assert(!one(&p,&out));
 reset();add(0,2);roots[0].chunk.va_block=NULL;assert(!one(&p,&out));
 // Mandatory PMA/allocation fallback still returns HOT and UNKNOWN roots.
 reset();add(0,2);mock_snapshot.access_bits[0]=1;
 uvm_mutex_lock(&p.lock);assert(pick_root_chunk_to_evict(&p)==&roots[0]);uvm_mutex_unlock(&p.lock);
 reset();add(0,2);vab_root_changed(&roots[0]);
 uvm_mutex_lock(&p.lock);assert(pick_root_chunk_to_evict(&p)==&roots[0]);uvm_mutex_unlock(&p.lock);
 // Free, UNUSED, and DISCARD lists retain their priority for mandatory demand.
 reset();add(0,2);add(1,1);add(2,0);p.free_list[0].first=&roots[3].chunk;
 uvm_mutex_lock(&p.lock);assert(pick_root_chunk_to_evict(&p)==&roots[3]);uvm_mutex_unlock(&p.lock);
 uvm_mutex_lock(&p.lock);assert(pick_root_chunk_to_evict(&p)==&roots[2]);uvm_mutex_unlock(&p.lock);
 uvm_mutex_lock(&p.lock);assert(pick_root_chunk_to_evict(&p)==&roots[1]);uvm_mutex_unlock(&p.lock);
 // Request-driven background reclaim preserves cached roots but mandatory
 // eviction still uses them; used HOT roots remain protected.
 reset();uvm_perf_prefetch_replacement_enable=true;
 add(0,2);p.free_list[0].first=&roots[3].chunk;
 assert(one(&p,&out)&&out==&roots[0]);
 assert(p.free_list[0].first==&roots[3].chunk);
 uvm_mutex_lock(&p.lock);assert(pick_root_chunk_to_evict(&p)==&roots[3]);uvm_mutex_unlock(&p.lock);
 reset();uvm_perf_prefetch_replacement_enable=true;add(0,2);mock_snapshot.access_bits[0]=1;
 assert(!one(&p,&out));
 // A batch scans the used list once and preserves oldest-first ordering.
 reset();for(unsigned i=0;i<6;++i){add(i,2);mock_recency[i]=mock_now-(i+1)*NSEC_PER_MSEC;}
 assert(eager_eviction_batch(&p,4,0)==4);
 assert(visits==6 && roots[5].chunk.claimed && roots[4].chunk.claimed &&
        roots[3].chunk.claimed && roots[2].chunk.claimed && !roots[1].chunk.claimed);
 // At realistic list sizes a full 16-victim batch still makes one scan.
 reset();for(unsigned i=0;i<256;++i){add(i,2);mock_recency[i]=mock_now-((i*73)%257+1)*NSEC_PER_MSEC;}
 uvm_eviction_candidates_t ranked={.limit=16};
 uvm_spin_lock(&p.list_lock);eviction_candidates_collect(&p,&mock_snapshot,&ranked);uvm_spin_unlock(&p.list_lock);
 assert(visits==256 && ranked.count==16);
 for(unsigned i=0;i<16;++i){
   unsigned best=256;
   for(unsigned j=0;j<256;++j)
     if(!roots[j].chunk.claimed && (best==256 || max(roots[j].vab_ready_ns,mock_recency[j])<max(roots[best].vab_ready_ns,mock_recency[best]) ||
        (max(roots[j].vab_ready_ns,mock_recency[j])==max(roots[best].vab_ready_ns,mock_recency[best]) && j>best)))best=j;
   assert(eager_eviction_one(&p,&out,&ranked)&&out==&roots[best]);
 }
 assert(visits==256);
 reset();for(unsigned i=0;i<256;++i)add(i,2);
 assert(eager_eviction_batch(&p,16,0)==16 && visits==256);
 printf("Selection scan regression: 256 visits for 16 victims (previously 3976).\n");
 // Larger configured batches collect another bounded group after draining.
 reset();for(unsigned i=0;i<256;++i)add(i,2);
 assert(eager_eviction_batch(&p,32,0)==32 && visits==256+240);
 // Candidate generations and current VAB evidence are revalidated before use.
 reset();for(unsigned i=0;i<4;++i)add(i,2);
 uvm_eviction_candidates_t candidates={.limit=4};
 uvm_spin_lock(&p.list_lock);eviction_candidates_collect(&p,&mock_snapshot,&candidates);uvm_spin_unlock(&p.list_lock);
 vab_root_changed(&roots[3]);mock_snapshot.access_bits[0]=1ULL<<2;
 assert(eager_eviction_one(&p,&out,&candidates)&&out==&roots[1]);
 assert(eager_eviction_one(&p,&out,&candidates)&&out==&roots[0]);
 assert(!eager_eviction_one(&p,&out,&candidates));
 // Roots claimed by another caller are skipped, without rescanning.
 reset();for(unsigned i=0;i<3;++i)add(i,2);
 candidates=(uvm_eviction_candidates_t){.limit=3};
 uvm_spin_lock(&p.list_lock);eviction_candidates_collect(&p,&mock_snapshot,&candidates);uvm_spin_unlock(&p.list_lock);
 roots[2].chunk.claimed=true;
 assert(eager_eviction_one(&p,&out,&candidates)&&out==&roots[1]);
 mock_recency_race=true;
 assert(!eager_eviction_one(&p,&out,&candidates));
 assert(!roots[0].chunk.claimed && visits==5);
 // A newer sample reranks candidates using current recency.
 reset();for(unsigned i=0;i<3;++i){add(i,2);mock_recency[i]=mock_now-(i+1)*NSEC_PER_MSEC;}
 candidates=(uvm_eviction_candidates_t){.limit=2};
 assert(eager_eviction_one(&p,&out,&candidates)&&out==&roots[2]);
 ++mock_snapshot.sampled_ns;++mock_now;mock_recency[1]=mock_now;
 assert(eager_eviction_one(&p,&out,&candidates)&&out==&roots[0]);
 // If the cached best root becomes hot, one refill finds an outside root.
 reset();for(unsigned i=0;i<3;++i)add(i,2);
 candidates=(uvm_eviction_candidates_t){.limit=1};
 uvm_spin_lock(&p.list_lock);eviction_candidates_collect(&p,&mock_snapshot,&candidates);uvm_spin_unlock(&p.list_lock);
 mock_snapshot.access_bits[0]=1ULL<<2;
 assert(eager_eviction_one(&p,&out,&candidates)&&out==&roots[1]);
 assert(candidates.refilled && p.vab_stats.background_hot_skips>0);
 // An invalidation before a successful pick is remembered until exhaustion.
 reset();for(unsigned i=0;i<3;++i)add(i,2);
 candidates=(uvm_eviction_candidates_t){.limit=2};
 uvm_spin_lock(&p.list_lock);eviction_candidates_collect(&p,&mock_snapshot,&candidates);uvm_spin_unlock(&p.list_lock);
 mock_snapshot.access_bits[0]=1ULL<<2;
 assert(eager_eviction_one(&p,&out,&candidates)&&out==&roots[1]);
 assert(eager_eviction_one(&p,&out,&candidates)&&out==&roots[0]);
 assert(candidates.refilled);
 // Expired snapshots never authorize a cached eviction.
 reset();add(0,2);candidates=(uvm_eviction_candidates_t){.limit=1};
 uvm_spin_lock(&p.list_lock);eviction_candidates_collect(&p,&mock_snapshot,&candidates);uvm_spin_unlock(&p.list_lock);
 mock_now+=301;
 assert(!eager_eviction_one(&p,&out,&candidates)&&!evictions);
 // Non-LRU selection stops scanning once the batch has enough cold roots.
 reset();for(unsigned i=0;i<6;++i)add(i,2);mock_snapshot.lru_enabled=false;
 assert(eager_eviction_batch(&p,2,0)==2&&visits==2);
 // Completed frees and failures are counted separately; batches stop at low.
 reset();for(unsigned i=0;i<5;++i)add(i,2);
 assert(eager_eviction_batch(&p,16,98)==2&&p.vab_stats.background_roots_freed==2);
 reset();for(unsigned i=0;i<5;++i)add(i,2);fail_at=2;
 assert(eager_eviction_batch(&p,16,95)==1&&p.vab_stats.background_failures==1);
 uvm_pmm_gpu_vab_stats_t stats;uvm_pmm_gpu_vab_stats(&p,&mock_snapshot,&stats);
 assert(stats.eligible_roots==3&&stats.used_cold==3);
 // Statistics expose idle, mature-but-unretired cohorts without changing policy.
 reset();
 p.eager_eviction.budget.cohorts[0]=(uvm_prefetch_cohort_t){.id=1,.tick=1,.blocks=32,.returns=16};
 p.eager_eviction.budget.cohorts[1]=(uvm_prefetch_cohort_t){.id=2,.tick=5,.blocks=4,.returns=1};
 mock_now=100*NSEC_PER_MSEC; // T=20ms: tick1 ready; tick5 still young.
 uvm_prefetch_budget_t before=p.eager_eviction.budget;
 uvm_pmm_gpu_vab_stats(&p,&mock_snapshot,&stats);
 assert(stats.prefetch_backoff_enabled==0 && stats.prefetch_recovery_enabled==0);
 assert(stats.prefetch_unretired_blocks==36 && stats.prefetch_unretired_returns==17);
 assert(stats.prefetch_ready_blocks==32 && stats.prefetch_ready_returns==16);
 assert(!memcmp(&before,&p.eager_eviction.budget,sizeof(before)));
 assert(diag_counts[UVM_PERF_DIAG_EVICTION_SELECT] > 0);
 assert(diag_counts[UVM_PERF_DIAG_EVICTION_SUBMIT] > 0);
 assert(diag_counts[UVM_PERF_DIAG_EVICTION_DRAIN] > 0);
 puts("Production VAB demand/background eviction policy and telemetry passed.");
 return 0;
}
'''

def main():
    source=(ROOT / 'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text()
    header=(ROOT / 'kernel-open/nvidia-uvm/uvm_vab.h').read_text().replace('#include "nvtypes.h"','')
    budget=(ROOT / 'kernel-open/nvidia-uvm/uvm_prefetch_budget.h').read_text().replace('#include "nvtypes.h"','').replace('static inline', 'static inline __attribute__((unused))')
    refault=(ROOT / 'kernel-open/nvidia-uvm/uvm_prefetch_refault.h').read_text().replace('#include "nvtypes.h"','').replace('static inline', 'static inline __attribute__((unused))')
    code=PRELUDE.replace('/* VAB classifier */',header+'\n#define UVM_ASSERT assert\n'+budget+refault)
    pmm_header = (ROOT / 'kernel-open/nvidia-uvm/uvm_pmm_gpu.h').read_text()
    stats_end = pmm_header.index('} uvm_pmm_gpu_vab_stats_t;') + len('} uvm_pmm_gpu_vab_stats_t;')
    stats = pmm_header[pmm_header.rfind('typedef struct', 0, stats_end):stats_end]
    code = code.replace('/* VAB stats */', stats)
    for parameter in ('churn_ms', 'return_percent', 'min_blocks', 'recovery_return_percent', 'bad_batches', 'good_batches'):
        code += re.search(r'static unsigned uvm_perf_speculative_prefetch_' + parameter + r' = \d+;', source).group(0) + '\n'
    for parameter in ('backoff_enable', 'recovery_enable', 'sparse_enable'):
        code += re.search(r'static bool uvm_perf_speculative_prefetch_' + parameter + r' = (?:true|false);', source).group(0) + '\n'
    code += function(source, 'uvm_pmm_gpu_prefetch_feedback_window_ns')
    code += function(source, 'speculative_prefetch_config')
    admission = (ROOT / 'kernel-open/nvidia-uvm/uvm_counter_admission.h').read_text()
    code += function(admission, 'uvm_adaptive_replacement_level')
    for name in ('prefetch_cached_free_bytes_locked', 'prefetch_free_bytes_locked',
                 'uvm_pmm_gpu_adaptive_admission_extra', 'adaptive_replacement_feedback'):
        code += function(source, name)
    start = source.index('#define UVM_EVICTION_CANDIDATES_MAX')
    end = source.index('// One used-list scan', start)
    code += source[start:end]
    for name in ('vab_root_changed','vab_root_state','vab_pick_allocated','get_first_allocated_chunk','vab_pick_root','pick_root_chunk_to_evict','eviction_candidate_newer','eviction_candidates_sift','eviction_candidates_collect','eviction_candidates_pick','eager_eviction_one','eager_eviction_batch','uvm_pmm_gpu_vab_stats'):
        code+=function(source,name)
    code+=SCENARIOS
    with tempfile.TemporaryDirectory(prefix='demand-first-eviction-') as directory:
        c=Path(directory)/'test.c';binary=Path(directory)/'test'
        c.write_text(code)
        subprocess.run(['cc','-std=gnu11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-g',str(c),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)

if __name__ == '__main__':
    main()
