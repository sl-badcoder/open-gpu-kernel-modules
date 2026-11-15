#!/usr/bin/env python3
"""Regression harness for access-counter demand service in CPU-preferred UVM.

This extracts the production service_va_block_locked() implementation.  The
old implementation let the normal residency selector choose CPU for a
CPU-preferred managed block, so the access-counter demand never advanced GPU
residency.  The harness models that selector and verifies the refault override
chooses GPU only for the accessed-page mask.
"""

import os
import pathlib
import subprocess
import tempfile


SOURCE = pathlib.Path(__file__).parents[2] / "kernel-open/nvidia-uvm/uvm_gpu_access_counters.c"


def extract(text, name):
    start = text.index(name)
    brace = text.index("{", start)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError(name)


PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include "uvm_prefetch_refault.h"
typedef int NV_STATUS; typedef int uvm_page_mask_t; typedef int uvm_page_index_t;
typedef int uvm_processor_id_t; typedef int uvm_processor_mask_t;
#define NV_OK 0
#define NV_ERR_INVALID_ADDRESS 7
#define UVM_ID_CPU 0
#define NUMA_NO_NODE (-1)
#define PAGES_PER_UVM_VA_BLOCK 4
#define UVM_FAULT_ACCESS_TYPE_PREFETCH 1
#define UVM_SERVICE_OPERATION_ACCESS_COUNTERS 3
#define UVM_ACCESS_COUNTER_ACTION_BATCH_CLEAR 1
#define UVM_ASSERT(x) assert(x)
#define READ_ONCE(x) (x)
#define uvm_id_value(x) (x)
#define uvm_processor_mask_test(m, id) ((*(m) >> uvm_id_value(id)) & 1)
static bool mask_test_and_set(int *m,int id) {bool old=(*m & (1<<id))!=0;*m|=1<<id;return old;}
#define uvm_processor_mask_test_and_set(m, id) mask_test_and_set((m),(id))
#define uvm_processor_mask_zero(m) (*(m) = 0)
#define uvm_processor_mask_clear(m,id) (*(m) &= ~(1 << uvm_id_value(id)))
#define uvm_processor_mask_empty(m) (*(m) == 0)
#define uvm_processor_mask_and(d,a,b) (*(d) = *(a) & *(b))
#define uvm_page_mask_empty(m) (*(m) == 0)
#define uvm_page_mask_zero(m) (*(m) = 0)
#define uvm_page_mask_set(m,i) (*(m) |= 1 << (i))
#define uvm_page_mask_copy(d,s) (*(d) = *(s))
#define uvm_page_mask_andnot(d,a,b) (*(d) &= ~(*(b)))
#define uvm_page_mask_and(d,a,b) ((*(d) = *(a) & *(b)) != 0)
#define for_each_id_in_mask(id,m) for ((id)=0; (id)<2; ++(id)) if (uvm_processor_mask_test((m),(id)))
#define for_each_va_block_page_in_mask(i,m,b) for ((i)=0; (i)<PAGES_PER_UVM_VA_BLOCK; ++(i)) if (*(m) & (1 << (i)))
typedef struct { int type; } uvm_perf_thrashing_hint_t;
#define UVM_PERF_THRASHING_HINT_TYPE_NONE 0
#define UVM_PERF_THRASHING_HINT_TYPE_THROTTLE 1
#define UVM_PERF_THRASHING_HINT_TYPE_PIN 2
typedef struct { int id; int parent; } uvm_gpu_t;
typedef struct { int cpu_access_counter_policy; } managed_range_t;
typedef struct { int id; int preferred_location; } uvm_va_policy_t;
typedef struct { int mm; struct { int vma; } hmm; struct { int pages_changed_residency; } make_resident; } block_ctx_t;
typedef struct { int new_residency; } per_mask_t;
typedef struct { block_ctx_t *block_context; int access_counters_buffer_index; int operation; int num_retries; int read_duplicate_count; int thrashing_pin_count; int thrashing_pin_mask; int resident_processors; per_mask_t per_processor_masks[3]; int update_processors; int region; int access_type[4]; } uvm_service_block_context_t;
typedef struct { int resident; int mapped; int lock; int managed_cpu; int start; managed_range_t *managed_range; } uvm_va_block_t;
typedef struct { int dummy; } uvm_va_block_retry_t;
static int gpu_resident_mask, cpu_resident_mask, hint_type, preferred;
static bool eligibility=true;
static int service_calls, service_destination_mask, service_pages;
static int refault_mode;
static bool uvm_pmm_gpu_refault_enabled(void) { return refault_mode; }
static bool uvm_cpu_block_policy_should_service_4k(managed_range_t *r, uvm_gpu_t *g) { return eligibility && r && r->cpu_access_counter_policy && g; }
static bool uvm_parent_gpu_is_coherent(int p) { (void)p; return false; }
static bool uvm_va_block_is_hmm(uvm_va_block_t *b) { return b->managed_cpu == 2; }
static bool uvm_cpu_block_policy_should_promote(uvm_va_block_t *b, uvm_gpu_t *g) { (void)b; (void)g; return false; }
static int uvm_hmm_find_vma(int mm, int *vma, NvU64 a) { (void)mm;(void)vma;(void)a; return NV_OK; }
static int uvm_hmm_find_policy_vma_and_outer(uvm_va_block_t *b, int *vma, int first, const uvm_va_policy_t **p, int *outer) { (void)b;(void)vma;(void)first;(void)p;*outer=PAGES_PER_UVM_VA_BLOCK;return NV_OK; }
static int uvm_va_block_region(int a, int b) { return (a << 16) | b; }
static bool UVM_ID_IS_CPU(int id) { return id == UVM_ID_CPU; }
static uvm_page_mask_t *uvm_va_block_resident_mask_get(uvm_va_block_t *b, int id, int n) { static int cpu, gpu; (void)b;(void)n; cpu = cpu_resident_mask; gpu = gpu_resident_mask; return id == UVM_ID_CPU ? &cpu : &gpu; }
static uvm_perf_thrashing_hint_t uvm_perf_thrashing_get_hint(uvm_va_block_t *b, block_ctx_t *c, NvU64 a, int p) { (void)b;(void)c;(void)a;(void)p; return (uvm_perf_thrashing_hint_t){hint_type}; }
static const uvm_va_policy_t *uvm_va_policy_get(uvm_va_block_t *b, NvU64 a) { (void)b;(void)a; static uvm_va_policy_t p; p.preferred_location=preferred;return &p; }
static int uvm_fault_access_type_mask_bit(int x) { return x; }
static int uvm_va_block_select_residency(uvm_va_block_t *b, block_ctx_t *c, int i, int p, int a, const uvm_va_policy_t *q, uvm_perf_thrashing_hint_t *h, int o, bool hm, bool *r) { (void)b;(void)c;(void)i;(void)p;(void)a;(void)q;(void)h;(void)o;(void)hm;*r=false; return UVM_ID_CPU; }
static NvU64 uvm_va_block_cpu_page_address(uvm_va_block_t *b, int i) { return (NvU64)(b->start + i * 4096); }
static void uvm_assert_mutex_locked(void *x) { (void)x; }
static bool uvm_page_mask_andnot_real(int *d, const int *a, const int *b) { *d = *a & ~*b; return *d != 0; }
#undef uvm_page_mask_andnot
#define uvm_page_mask_andnot(d,a,b) uvm_page_mask_andnot_real((d),(a),(b))
static NV_STATUS uvm_va_block_service_locked(uvm_gpu_t *g, uvm_va_block_t *b, uvm_va_block_retry_t *r, uvm_service_block_context_t *c) { (void)b;(void)r; service_calls++; service_destination_mask = c->resident_processors; service_pages = c->per_processor_masks[uvm_id_value(g->id)].new_residency; gpu_resident_mask |= service_pages;cpu_resident_mask &= ~service_pages;b->resident |= 1<<g->id; c->block_context->make_resident.pages_changed_residency = service_pages; return NV_OK; }
static void uvm_cpu_block_policy_record_promotion(void) {}
static bool uvm_page_mask_empty_real(const int *m) { return *m == 0; }
#undef uvm_page_mask_empty
#define uvm_page_mask_empty(m) uvm_page_mask_empty_real((m))
#define UVM_VA_BLOCK_RETRY_LOCKED(b,r,expr) (expr)
'''


TESTS = r'''
static void init_ctx(uvm_va_block_t *b, uvm_service_block_context_t *c, block_ctx_t *bc) {
 memset(b,0,sizeof(*b));memset(c,0,sizeof(*c));memset(bc,0,sizeof(*bc));c->block_context=bc;
 b->mapped=2;b->resident=1;b->start=0;b->managed_cpu=1;
 static managed_range_t range={1};b->managed_range=&range;
 gpu_resident_mask=0;cpu_resident_mask=0xf;service_calls=0;service_pages=0;
 hint_type=UVM_PERF_THRASHING_HINT_TYPE_NONE;preferred=UVM_ID_CPU;eligibility=true;
}
int main(void) {
 uvm_gpu_t gpu={.id=1};uvm_va_block_t b;uvm_service_block_context_t c;block_ctx_t bc;
 uvm_va_block_retry_t retry;bool migrated=false;int accessed;
 (void)uvm_pmm_gpu_refault_enabled;(void)uvm_cpu_block_policy_should_service_4k;(void)UVM_ID_IS_CPU;
 init_ctx(&b,&c,&bc);accessed=5;refault_mode=0;
 assert(service_va_block_locked(&gpu,&b,&retry,&c,&accessed,&migrated)==NV_OK);
 assert(!service_calls && !gpu_resident_mask && cpu_resident_mask==0xf);
 // The actual production loop must migrate exactly demanded pages, never the
 // untouched pages2/4 of this toy block. Full-block promotion remains disabled.
 init_ctx(&b,&c,&bc);accessed=5;refault_mode=1;
 assert(service_va_block_locked(&gpu,&b,&retry,&c,&accessed,&migrated)==NV_OK);
 assert(service_calls==1 && service_pages==5 && gpu_resident_mask==5 && cpu_resident_mask==10);
 assert(service_destination_mask==2 && migrated);
 // Actual resulting residency feeds the production distance/aging policy.
 uvm_prefetch_refault_t history={0};uvm_prefetch_refault_block_t shadow={0};
 uvm_prefetch_refault_config_t cfg={.capacity_bytes=64ULL<<20,.short_distance_percent=5,.max_neighbors=4};
 NvU64 moved=(NvU64)__builtin_popcount((unsigned)(0xf & gpu_resident_mask))*4096;
 uvm_prefetch_refault_demand(&history,&cfg,&shadow,7,0,moved,0,0,true,false);
 assert(history.demand_bytes==8192);
 // Neighbor copies are excluded from the demand clock.
 uvm_prefetch_refault_population(&history,&cfg,7,2ULL<<20);
 assert(history.demand_bytes==8192);
 uvm_prefetch_refault_evict(&history,&shadow);
 uvm_prefetch_refault_demand(&history,&cfg,NULL,8,64ULL<<20,4ULL<<20,0,1,false,false);
 uvm_prefetch_refault_demand(&history,&cfg,&shadow,7,0,moved,0,2,true,true);
 assert(history.refaults==1 && history.short_refaults==0 && history.last_distance==(4ULL<<20));
 uvm_prefetch_refault_tile(&history,7,0)->window=0;
 assert(!uvm_prefetch_refault_neighbors(&history,&cfg,7,0));
 uvm_prefetch_refault_demand(&history,&cfg,NULL,8,64ULL<<20,16ULL<<20,0,3,false,false);
 assert(uvm_prefetch_refault_neighbors(&history,&cfg,7,0)==1);
 // Native thrashing choices retain precedence over the experiment.
 for(int hint=UVM_PERF_THRASHING_HINT_TYPE_THROTTLE;hint<=UVM_PERF_THRASHING_HINT_TYPE_PIN;hint++) {
  init_ctx(&b,&c,&bc);accessed=5;hint_type=hint;
  assert(service_va_block_locked(&gpu,&b,&retry,&c,&accessed,&migrated)==NV_OK);
  assert(!gpu_resident_mask && cpu_resident_mask==0xf);
  assert(c.thrashing_pin_count==(hint==UVM_PERF_THRASHING_HINT_TYPE_PIN?2:0));
 }
 // HMM, unsupported policies, non-CPU preferences, and absent managed ranges
 // continue to use the ordinary selector, with no forced GPU residency.
 for(int excluded=0;excluded<4;excluded++) {
  init_ctx(&b,&c,&bc);accessed=5;
  if(excluded==0)b.managed_cpu=2;
  if(excluded==1)eligibility=false;
  if(excluded==2)preferred=2;
  if(excluded==3)b.managed_range=NULL;
  assert(service_va_block_locked(&gpu,&b,&retry,&c,&accessed,&migrated)==NV_OK);
  assert(!gpu_resident_mask && cpu_resident_mask==0xf);
 }
 // A stale notification for a GPU-resident page must not migrate it twice.
 init_ctx(&b,&c,&bc);accessed=3;gpu_resident_mask=1;cpu_resident_mask=14;b.resident=3;
 assert(service_va_block_locked(&gpu,&b,&retry,&c,&accessed,&migrated)==NV_OK);
 assert(service_pages==2 && gpu_resident_mask==3 && cpu_resident_mask==12);
 return 0;
}
'''



def main():
    function = extract(pathlib.Path(os.environ.get("REFAULT_TEST_SOURCE", str(SOURCE))).read_text(), "static NV_STATUS service_va_block_locked")
    source = PRELUDE + function + TESTS
    with tempfile.TemporaryDirectory(prefix="refault-demand-service-") as d:
        cfile = pathlib.Path(d) / "harness.c"
        binary = pathlib.Path(d) / "harness"
        cfile.write_text(source)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", "-I", str(SOURCE.parents[1] / "common/inc"), "-I", str(SOURCE.parent), str(cfile), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
    print("Production demand service: accessed-page migration, policy exclusions, distance clock and aging passed.")
