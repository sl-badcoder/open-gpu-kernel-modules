#!/usr/bin/env python3
"""Run production root invalidation and nonblocking completion polling with mocks."""
import re
import subprocess
import tempfile
from pathlib import Path
ROOT=Path(__file__).parents[2]

def function(text,name):
    m=re.search(r'^(?:static [^\n]*|void )\b'+name+r'\(',text,re.M)
    if not m: raise ValueError(name)
    end=text.index('{',m.start())+1;depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[m.start():end]

PRELUDE=r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
typedef uint64_t NvU64;
#define NV_OK 0
#define NSEC_PER_MSEC 1000000ULL
#define UVM_CHUNK_SIZE_MAX (2ULL<<20)
#define UVM_PMM_GPU_CHUNK_STATE_ALLOCATED 1
#define min_t(t,a,b) ((a)<(b)?(a):(b))
typedef struct {int held;} lock_t;
typedef struct block {lock_t lock;int tracker,refs;bool hmm;NvU64 start,size;}uvm_va_block_t;
typedef struct chunk {int state;bool user,eviction;struct chunk *parent;uvm_va_block_t *va_block;}uvm_gpu_chunk_t;
typedef struct {uvm_gpu_chunk_t chunk;NvU64 vab_generation,vab_ready_generation,vab_ready_ns;}uvm_gpu_root_chunk_t;
typedef struct {lock_t list_lock;struct {uvm_gpu_root_chunk_t *array;size_t count;}root_chunks;size_t vab_scan_cursor;NvU64 vab_scan_next_ns;struct {unsigned readiness_pending;}vab_stats;}uvm_pmm_gpu_t;
typedef struct {int unused;}uvm_parent_gpu_t;
typedef struct {uvm_pmm_gpu_t pmm;uvm_parent_gpu_t *parent;}uvm_gpu_t;
static struct {struct {int lock;}pm;}g_uvm_global;
static uvm_gpu_t g;
static uvm_gpu_root_chunk_t roots[1025];
static uvm_va_block_t b,other;
static uvm_gpu_chunk_t *lookup;
static NvU64 clock_now;
static bool requested,pm_ok,try_ok;
static int pm_held,tracker_result,mutate;
static unsigned queries,retains,releases;
static uvm_gpu_root_chunk_t *changing_root;
#define NV_GETTIME() (clock_now++)
static uvm_gpu_t *uvm_pmm_to_gpu(uvm_pmm_gpu_t *p){return (uvm_gpu_t *)((char *)p - offsetof(uvm_gpu_t,pmm));}
static bool uvm_gpu_vab_eviction_requested(uvm_parent_gpu_t *p){(void)p;return requested;}
static bool uvm_down_read_trylock(int *p){(void)p;if(!pm_ok)return false;assert(!pm_held);pm_held=1;return true;}
static void uvm_up_read(int *p){(void)p;assert(pm_held);pm_held=0;}
static void uvm_spin_lock(lock_t *l){assert(!l->held);l->held=1;}
static void uvm_spin_unlock(lock_t *l){assert(l->held);l->held=0;}
static bool uvm_gpu_chunk_is_user(uvm_gpu_chunk_t *c){return c->user;}
static bool chunk_is_in_eviction(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *c){assert(p->list_lock.held);return c->eviction;}
static void uvm_va_block_retain(uvm_va_block_t *v){assert(g.pmm.list_lock.held);++v->refs;++retains;}
static void uvm_va_block_release(uvm_va_block_t *v){assert(!v->lock.held&&!g.pmm.list_lock.held);--v->refs;++releases;}
static bool uvm_mutex_trylock(lock_t *l){assert(!g.pmm.list_lock.held&&!l->held);if(!try_ok)return false;l->held=1;return true;}
static void uvm_mutex_unlock(lock_t *l){assert(l->held);l->held=0;}
static bool uvm_va_block_is_hmm(uvm_va_block_t *v){return v->hmm;}
static int uvm_tracker_query(int *t){
 uvm_va_block_t *v=(uvm_va_block_t *)((char *)t - offsetof(uvm_va_block_t,tracker));
 assert(v->lock.held&&!g.pmm.list_lock.held);++queries;
 if(mutate==1)++changing_root->vab_generation;
 if(mutate==2)changing_root->chunk.va_block=&other;
 return tracker_result;
}
static NvU64 uvm_va_block_size(uvm_va_block_t *v){return v->size;}
static uvm_gpu_chunk_t *uvm_va_block_lookup_gpu_chunk(uvm_va_block_t *v,uvm_gpu_t *gpu,NvU64 a){assert(v->lock.held&&gpu==&g&&a==v->start);return lookup;}
static uvm_gpu_root_chunk_t *root_chunk_from_chunk(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *c){assert(p==&g.pmm);return (uvm_gpu_root_chunk_t *)c;}
static void uvm_assert_mutex_locked(lock_t *l){assert(l->held);}
'''
CASES=r'''
static void reset(void){
 memset(&g,0,sizeof(g));memset(roots,0,sizeof(roots));memset(&b,0,sizeof(b));memset(&other,0,sizeof(other));
 b.size=UVM_CHUNK_SIZE_MAX;g.pmm.root_chunks.array=roots;g.pmm.root_chunks.count=1;
 roots[0].chunk=(uvm_gpu_chunk_t){.state=1,.user=true,.va_block=&b};roots[0].vab_generation=7;
 lookup=&roots[0].chunk;clock_now=1000000000ULL;requested=pm_ok=try_ok=true;
 pm_held=tracker_result=mutate=0;queries=retains=releases=0;changing_root=&roots[0];
}
static void poll(void){clock_now+=100000000;vab_prepare_roots(&g.pmm);assert(!pm_held&&!g.pmm.list_lock.held&&retains==releases&&!b.refs);}
static void invalidate(void){b.lock.held=1;uvm_pmm_gpu_vab_invalidate_block(&b,&g);b.lock.held=0;}
int main(void){
 reset();requested=false;poll();assert(!queries&&!retains);
 requested=true;pm_ok=false;poll();assert(!queries&&!retains);
 reset();tracker_result=1;poll();assert(queries==1&&!roots[0].vab_ready_ns&&g.pmm.vab_stats.readiness_pending==1);
 tracker_result=0;NvU64 before=clock_now;poll();assert(roots[0].vab_ready_generation==7&&roots[0].vab_ready_ns>before);
 unsigned prior=queries;poll();assert(queries==prior); // ready roots need no repeated GPU polling
 invalidate();assert(roots[0].vab_generation==8&&!roots[0].vab_ready_ns&&!roots[0].vab_ready_generation);
 poll();assert(roots[0].vab_ready_generation==8&&roots[0].vab_ready_ns>before);
 reset();try_ok=false;poll();assert(!queries&&!roots[0].vab_ready_ns&&retains==1&&releases==1);
 reset();mutate=1;poll();assert(!roots[0].vab_ready_ns&&roots[0].vab_generation==8);
 reset();mutate=2;poll();assert(!roots[0].vab_ready_ns&&roots[0].chunk.va_block==&other);
 reset();roots[0].vab_generation=0;poll();assert(!roots[0].vab_ready_ns);
 reset();roots[0].chunk.state=0;poll();assert(!queries&&!retains);
 reset();roots[0].chunk.user=false;poll();assert(!queries&&!retains);
 reset();roots[0].chunk.eviction=true;poll();assert(!queries&&!retains);
 reset();roots[0].chunk.va_block=NULL;poll();assert(!queries&&!retains);
 reset();b.hmm=true;poll();assert(!queries&&!roots[0].vab_ready_ns);
 // No generation mutation for unsupported HMM/split/small/absent roots.
 reset();roots[0].vab_ready_ns=123;b.hmm=true;invalidate();assert(roots[0].vab_generation==7&&roots[0].vab_ready_ns==123);
 b.hmm=false;b.size=4096;invalidate();assert(roots[0].vab_generation==7);
 b.size=UVM_CHUNK_SIZE_MAX;roots[0].chunk.parent=&roots[1].chunk;invalidate();assert(roots[0].vab_generation==7);
 roots[0].chunk.parent=NULL;lookup=NULL;invalidate();assert(roots[0].vab_generation==7);
 lookup=&roots[0].chunk;invalidate();assert(roots[0].vab_generation==8&&!roots[0].vab_ready_ns);
 roots[0].vab_generation=UINT64_MAX;vab_root_changed(&roots[0]);assert(roots[0].vab_generation==1);
 // One pass visits at most 1024 slots; the cursor advances fairly.
 reset();memset(roots,0,sizeof(roots));g.pmm.root_chunks.count=1025;
 roots[1024].chunk=(uvm_gpu_chunk_t){.state=1,.user=true,.va_block=&b};roots[1024].vab_generation=7;
 poll();assert(!queries&&g.pmm.vab_scan_cursor==1024&&!roots[1024].vab_ready_ns);
 poll();assert(queries==1&&roots[1024].vab_ready_ns);
 puts("VAB production readiness: completion, invalidation, races, ownership, bounded polling and balanced references passed.");
 return 0;
}
'''
def main():
    text=(ROOT/'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text()
    code=PRELUDE+''.join(function(text,n) for n in ['vab_root_changed','vab_prepare_roots','uvm_pmm_gpu_vab_invalidate_block'])+CASES
    with tempfile.TemporaryDirectory(prefix='vab-ready-') as d:
        c=Path(d)/'test.c';out=Path(d)/'test';c.write_text(code)
        subprocess.run(['cc','-std=c11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(c),'-o',str(out)],check=True)
        subprocess.run([str(out)],check=True)
if __name__=='__main__':main()
