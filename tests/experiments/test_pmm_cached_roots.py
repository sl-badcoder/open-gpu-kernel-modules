#!/usr/bin/env python3
"""Test production whole-user-root cache accounting through allocator states."""
from pathlib import Path
import subprocess
import tempfile
from test_demand_first_eviction import function

ROOT=Path(__file__).parents[2]
PRELUDE=r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef uint64_t NvU64; typedef unsigned NvU32;
typedef unsigned uvm_pmm_gpu_chunk_state_t;
typedef unsigned uvm_pmm_gpu_memory_type_t;
#define UVM_PMM_GPU_CHUNK_STATE_FREE 0
#define UVM_PMM_GPU_CHUNK_STATE_ALLOCATED 1
#define UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED 2
#define UVM_PMM_GPU_CHUNK_STATE_IS_SPLIT 3
#define UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED 4
#define UVM_PMM_GPU_MEMORY_TYPE_USER 0
#define UVM_PMM_GPU_MEMORY_TYPE_KERNEL 1
#define UVM_CHUNK_SIZE_MAX (2ULL<<20)
#define UVM_ASSERT assert
static void assert_msg(bool cond,const char *fmt,...) {(void)fmt;assert(cond);}
#define UVM_ASSERT_MSG(c,...) assert_msg(c,__VA_ARGS__)
struct list_head {bool linked;};
static bool list_empty(struct list_head *l) {return !l->linked;}
static void list_del_init(struct list_head *l) {l->linked=false;}
struct suballoc {unsigned pinned_leaf_chunks,allocated;};
typedef struct chunk {unsigned state,type;struct chunk *parent;struct suballoc *suballoc;
 void *va_block,*va_space;bool is_zero,in_eviction;NvU64 address;struct list_head list;} uvm_gpu_chunk_t;
typedef struct {uvm_gpu_chunk_t chunk;int tracker;} uvm_gpu_root_chunk_t;
typedef struct {int list_lock,lock,pma_lock;struct {NvU64 cached_free_root_count;long pinned_count;unsigned in_eviction_count;} root_chunks;} uvm_pmm_gpu_t;
typedef struct {int unused;} uvm_gpu_t;
static uvm_gpu_t gpu;
static uvm_gpu_t *uvm_pmm_to_gpu(uvm_pmm_gpu_t *p) {(void)p;return &gpu;}
static const char *uvm_gpu_name(uvm_gpu_t *g) {(void)g;return "GPU";}
static const char *uvm_pmm_gpu_chunk_state_string(unsigned s) {(void)s;return "state";}
static bool chunk_is_root_chunk(uvm_gpu_chunk_t *c) {return !c->parent;}
static bool uvm_gpu_chunk_is_user(uvm_gpu_chunk_t *c) {return c->type==UVM_PMM_GPU_MEMORY_TYPE_USER;}
static uvm_gpu_root_chunk_t *root_chunk_from_chunk(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *c) {(void)p;while(c->parent)c=c->parent;return (uvm_gpu_root_chunk_t *)c;}
static bool chunk_is_root_chunk_pinned(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *c) {uvm_gpu_chunk_t *r=&root_chunk_from_chunk(p,c)->chunk;return r->state==2||(r->suballoc&&r->suballoc->pinned_leaf_chunks);}
static void uvm_assert_spinlock_locked(int *l) {assert(*l);}
static void uvm_assert_mutex_locked(int *l) {assert(*l);}
static void uvm_assert_rwsem_locked(int *l) {assert(*l);}
static void uvm_spin_lock(int *l) {assert(!*l);*l=1;}
static void uvm_spin_unlock(int *l) {assert(*l);*l=0;}
static void chunk_update_lists_locked(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *c) {assert(p->list_lock);c->list.linked=c->state==0;}
static bool chunk_is_evictable(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *c) {(void)p;return !c->in_eviction;}
static void uvm_gpu_chunk_set_in_eviction(uvm_gpu_chunk_t *c,bool yes) {c->in_eviction=yes;}
static NvU64 uvm_gpu_chunk_get_size(uvm_gpu_chunk_t *c) {(void)c;return UVM_CHUNK_SIZE_MAX;}
static bool root_chunk_has_elevated_page(uvm_pmm_gpu_t *p,uvm_gpu_root_chunk_t *r) {(void)p;(void)r;return false;}
static void uvm_pmm_gpu_root_chunk_lock(uvm_pmm_gpu_t *p,uvm_gpu_root_chunk_t *r) {(void)p;(void)r;}
static void uvm_pmm_gpu_root_chunk_unlock(uvm_pmm_gpu_t *p,uvm_gpu_root_chunk_t *r) {(void)p;(void)r;}
static void uvm_tracker_init(int *t) {*t=1;}
static void vab_root_changed(uvm_gpu_root_chunk_t *r) {(void)r;}
'''
CASES=r'''
int main(void) {
 uvm_pmm_gpu_t p={.lock=1,.pma_lock=1};uvm_gpu_root_chunk_t r={.chunk={.state=4}};
 // A PMA batch adds unclaimed user roots to the reusable cache.
 init_root_chunk(&p,0,&r,0,false);assert(p.root_chunks.cached_free_root_count==1);
 p.list_lock=1;chunk_pin(&p,&r.chunk);assert(!p.root_chunks.cached_free_root_count&&p.root_chunks.pinned_count==1);
 chunk_unpin(&p,&r.chunk,1);assert(!p.root_chunks.cached_free_root_count);
 chunk_free_locked(&p,&r.chunk);assert(p.root_chunks.cached_free_root_count==1);
 // A root selected for mandatory cache draining loses credit immediately,
 // before its later FREE->PINNED transition. Never debit it twice.
 r.chunk.list.linked=true;chunk_start_eviction(&p,&r.chunk);
 assert(!p.root_chunks.cached_free_root_count&&r.chunk.in_eviction);
 chunk_pin(&p,&r.chunk);assert(!p.root_chunks.cached_free_root_count);
 chunk_unpin(&p,&r.chunk,4);assert(!p.root_chunks.cached_free_root_count);
 r.chunk.in_eviction=false;p.list_lock=0;
 // Kernel roots cannot supply user-population cache credit.
 init_root_chunk(&p,1,&r,0,true);assert(!p.root_chunks.cached_free_root_count);
 p.list_lock=1;chunk_pin(&p,&r.chunk);chunk_unpin(&p,&r.chunk,0);assert(!p.root_chunks.cached_free_root_count);
 // A directly freed pinned user root restores exactly one credit.
 r.chunk.type=0; r.chunk.state=2;p.root_chunks.pinned_count=1;
 chunk_free_locked(&p,&r.chunk);assert(p.root_chunks.cached_free_root_count==1&&!p.root_chunks.pinned_count);
 // Free subchunks are not additional whole-root credits.
 struct suballoc sub={.allocated=2};uvm_gpu_chunk_t child={.state=1,.type=0,.parent=&r.chunk};
 r.chunk.state=3;r.chunk.suballoc=&sub;p.root_chunks.cached_free_root_count=0;
 chunk_free_locked(&p,&child);assert(!p.root_chunks.cached_free_root_count&&sub.allocated==1);
 puts("Production PMM cached-root accounting: init, claim, free, eviction, kernel and subchunk transitions passed.");
}
'''
def main():
 source=(ROOT/'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text();code=PRELUDE
 for name in ('chunk_pin','chunk_unpin','chunk_start_eviction','init_root_chunk','chunk_free_locked'):
  code+=function(source,name)
 code+=CASES
 with tempfile.TemporaryDirectory(prefix='cached-roots-') as d:
  c=Path(d)/'test.c';exe=Path(d)/'test';c.write_text(code)
  subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(c),'-o',str(exe)],check=True)
  subprocess.run([str(exe)],check=True)
if __name__=='__main__':main()
