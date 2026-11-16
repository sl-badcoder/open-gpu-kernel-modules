#!/usr/bin/env python3
"""Run actual GPU-state teardown to prevent live-byte leakage between workloads."""
from pathlib import Path
import subprocess
import tempfile
from test_demand_first_eviction import function
ROOT=Path(__file__).parents[2]
PRELUDE=r'''
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
typedef uint64_t NvU64;typedef unsigned NvU32;typedef int uvm_gpu_id_t;
#define PAGE_SIZE 4096ULL
#define UVM_ASSERT assert
'''
STUBS=r'''
typedef NvU64 uvm_page_mask_t;
typedef struct {int unused;} uvm_gpu_chunk_t;
typedef struct {struct {uvm_prefetch_replacement_state_t replacement;} eager_eviction;int list_lock;} uvm_pmm_gpu_t;
typedef struct {uvm_gpu_id_t id;uvm_pmm_gpu_t pmm;} uvm_gpu_t;
typedef struct {uvm_page_mask_t resident,replacement_live;uvm_gpu_chunk_t **chunks;} uvm_va_block_gpu_state_t;
typedef struct {int unused;} uvm_va_space_t;
typedef struct {int unused;} uvm_gpu_va_space_t;
typedef struct {int unused;} uvm_va_block_context_t;
typedef struct {int lock,mapped,resident,evicted_gpus,tracker;uvm_va_block_gpu_state_t *gpus[1];} uvm_va_block_t;
static uvm_gpu_t gpu;
static bool uvm_pmm_gpu_prefetch_turnover_enabled(void) {return true;}
static int g_uvm_va_block_gpu_state_cache,destructions,chunk_frees;
static void uvm_assert_mutex_locked(int *l) {assert(*l);}
static void uvm_spin_lock(int *l) {assert(!*l);*l=1;}
static void uvm_spin_unlock(int *l) {assert(*l);*l=0;}
static uvm_va_block_gpu_state_t *uvm_va_block_gpu_state_get(uvm_va_block_t *b,int id) {(void)id;assert(b->lock);return b->gpus[0];}
static unsigned uvm_page_mask_weight(uvm_page_mask_t *p) {return __builtin_popcountll(*p);}
static void uvm_page_mask_zero(uvm_page_mask_t *p) {*p=0;}
static uvm_va_space_t *uvm_va_block_get_va_space(uvm_va_block_t *b) {(void)b;return 0;}
static uvm_gpu_t *uvm_gpu_get(int id) {assert(id==0);return &gpu;}
static uvm_gpu_va_space_t *uvm_gpu_va_space_get(uvm_va_space_t *v,uvm_gpu_t *g) {(void)v;(void)g;return 0;}
static void uvm_va_block_remove_gpu_va_space(uvm_va_block_t *b,uvm_gpu_va_space_t *s,uvm_va_block_context_t *c) {(void)b;(void)s;(void)c;}
static bool uvm_processor_mask_test(int *m,int id) {(void)id;return *m;}
static void uvm_processor_mask_clear(int *m,int id) {(void)id;*m=0;}
static bool block_check_processor_not_mapped(uvm_va_block_t *b,uvm_va_block_context_t *c,int id) {(void)b;(void)c;(void)id;return true;}
static void update_read_duplicated_pages_mask(uvm_va_block_t *b,int id,uvm_va_block_gpu_state_t *s) {(void)b;(void)id;(void)s;}
static void block_clear_resident_processor(uvm_va_block_t *b,int id) {(void)id;b->resident=0;}
static unsigned block_num_gpu_chunks(uvm_va_block_t *b,uvm_gpu_t *g) {(void)b;(void)g;return 1;}
static void uvm_mmu_chunk_unmap(uvm_gpu_chunk_t *c,int *t) {(void)c;(void)t;}
static void uvm_pmm_gpu_free(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *c,int *t) {(void)p;(void)c;(void)t;++chunk_frees;}
static void uvm_kvfree(void *p) {assert(p);}
static void uvm_tracker_wait(int *t) {*t=1;}
static void block_gpu_unmap_phys_all_cpu_pages(uvm_va_block_t *b,uvm_gpu_t *g) {(void)g;assert(b->tracker);}
static void kmem_cache_free(int cache,uvm_va_block_gpu_state_t *s) {(void)cache;assert(!s->replacement_live);++destructions;}
static unsigned uvm_id_gpu_index(int id) {return (unsigned)id;}
'''
CASES=r'''
int main(void) {
 uvm_gpu_chunk_t c={0};uvm_gpu_chunk_t *chunks[]={&c};uvm_va_block_context_t ctx={0};
 uvm_va_block_gpu_state_t a={.resident=7,.replacement_live=7,.chunks=chunks};
 uvm_va_block_gpu_state_t b={.resident=3,.replacement_live=3,.chunks=chunks};
 uvm_va_block_t ba={.lock=1,.resident=1,.gpus={&a}},bb={.lock=1,.resident=1,.gpus={&b}};
 uvm_prefetch_replacement_state_t *s=&gpu.pmm.eager_eviction.replacement;
 s->resident_bytes=5*PAGE_SIZE;s->population_bytes=8*PAGE_SIZE;
 s->request_stride=4;s->epoch_stamp=123;s->epoch_roots_used=8;
 s->window_population_bytes=64ULL<<20;s->last_population_bytes=64ULL<<20;
 block_destroy_gpu_state(&ba,&ctx,0);
 assert(!ba.gpus[0]&&s->resident_bytes==2*PAGE_SIZE&&s->request_stride==4);
 block_destroy_gpu_state(&bb,&ctx,0);
 assert(!bb.gpus[0]&&s->resident_bytes==0&&s->request_stride==1);
 assert(!s->epoch_stamp&&!s->epoch_roots_used&&!s->window_population_bytes&&!s->last_population_bytes);
 assert(s->population_bytes==8*PAGE_SIZE&&s->evicted_bytes==0&&destructions==2&&chunk_frees==2);
 // A subsequent workload starts with a clean live gauge, not the previous total.
 s->resident_bytes=PAGE_SIZE;uvm_va_block_gpu_state_t next={.replacement_live=1};
 uvm_va_block_t bn={.lock=1,.gpus={&next}};block_destroy_gpu_state(&bn,&ctx,0);
 assert(s->resident_bytes==0&&destructions==3);
 // The final state can already have zero live pages after prior evictions.
 // Its destruction must still discard stale control observations.
 s->request_stride=4;s->window_population_bytes=123;
 uvm_va_block_gpu_state_t empty={0};uvm_va_block_t be={.lock=1,.gpus={&empty}};
 block_destroy_gpu_state(&be,&ctx,0);
 assert(s->request_stride==1&&!s->window_population_bytes&&destructions==4);
 puts("Production GPU-state teardown: live bytes removed once, shared-GPU state preserved, controller reset between workloads passed.");
}
'''
def main():
 cpu=(ROOT/'kernel-open/nvidia-uvm/uvm_cpu_block_policy.c').read_text()
 block=(ROOT/'kernel-open/nvidia-uvm/uvm_va_block.c').read_text()
 code=PRELUDE+(ROOT/'kernel-open/nvidia-uvm/uvm_prefetch_replacement.h').read_text()+STUBS
 code+=function(cpu,'uvm_cpu_block_policy_forget_population')
 code+=function(block,'block_destroy_gpu_state')+CASES
 with tempfile.TemporaryDirectory(prefix='turnover-lifecycle-') as d:
  c=Path(d)/'test.c';exe=Path(d)/'test';c.write_text(code)
  subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-Wno-unused-function','-fsanitize=address,undefined',str(c),'-o',str(exe)],check=True)
  subprocess.run([str(exe)],check=True)
if __name__=='__main__':main()
