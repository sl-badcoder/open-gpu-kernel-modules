#!/usr/bin/env python3
"""Regression checks for the behavior-preserving 615 access-counter port.

The harnesses compile extracted production functions/snippets. They do not
pretend to build the Linux module; their purpose is to catch accidental loss
of the old CPU-preferred mapping and full-block promotion behavior.
"""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).parents[2]

def extract_function(source, name):
    start = source.index(name)
    brace = source.index("{", start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{": depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0: return source[start:i + 1]
    raise ValueError(name)

def extract_block(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{": depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0: return source[start:i + 1]
    raise ValueError(marker)

def run_policy_harness():
    source = (ROOT / "kernel-open/nvidia-uvm/uvm_cpu_block_policy.c").read_text()
    enabled = extract_function(source, "bool uvm_cpu_block_policy_enabled")
    supported = extract_function(source, "static bool gpu_supported")
    init_range = extract_function(source, "void uvm_cpu_block_policy_init_range")
    add_accessed = extract_function(source, "bool uvm_cpu_block_policy_should_add_accessed_by")
    service4k = extract_function(source, "bool uvm_cpu_block_policy_should_service_4k")
    prelude = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
typedef unsigned NvU32; typedef int uvm_processor_id_t;
#define UVM_ID_CPU 0
#define UVM_ID_IS_CPU(x) ((x)==UVM_ID_CPU)
#define NUMA_NO_NODE (-1)
#define UVM_LOCK_ORDER_LEAF 0
#define READ_ONCE(x) (x)
#define uvm_id_value(x) (x)
#define UVM_ASSERT(x) assert(x)
typedef struct gpu_parent { bool access_counters_supported, is_integrated_gpu; } gpu_parent;
typedef struct gpu { gpu_parent *parent; int id; } uvm_gpu_t;
typedef struct va_space { int lock; unsigned accessible_from[4]; uvm_gpu_t *gpus[4]; int gpu_count; } uvm_va_space_t;
typedef struct block { unsigned gpu_prefetch_started, gpu_prefetch_retry; struct block *next; } uvm_va_block_t;
typedef struct managed { struct { uvm_va_space_t *va_space; } va_range; bool cpu_access_counter_policy; struct { int preferred_location, preferred_nid; unsigned accessed_by; } policy; int cpu_prefetch_lock; uvm_va_block_t *blocks; } uvm_va_range_managed_t;
struct { bool conf_computing_enabled; } g_uvm_global;
static unsigned uvm_cpu_preferred_blocks_enable = 1;
static void uvm_assert_rwsem_locked_write(int *x) {(void)x;} static void uvm_assert_rwsem_locked(int *x) {(void)x;}
static void uvm_processor_mask_clear_atomic(unsigned *m,int id){*m&=~(1u<<id);}
#define for_each_va_space_gpu(g,v) for (int _i=0; _i<(v)->gpu_count && ((g)=(v)->gpus[_i],1); ++_i)
#define for_each_va_block_in_va_range(r,b) for ((b)=(r)->blocks; (b); (b)=(b)->next)
#define uvm_processor_mask_test(m,id) (((*(m))>>id)&1)
#define uvm_processor_mask_set(m,id) (*(m)|=1u<<id)
#define UVM_ID_IS_CPU(x) ((x)==UVM_ID_CPU)
'''
    code = prelude + enabled + supported + init_range + add_accessed + service4k + r'''
int main(void) {
    gpu_parent p={1,0}; uvm_gpu_t g={&p,1}; uvm_va_space_t v={0,{0}, {&g},1};
    v.accessible_from[0] = 1u << 1; uvm_va_range_managed_t r={.va_range={&v}};
    uvm_cpu_block_policy_init_range(&r);
    assert(r.cpu_access_counter_policy && r.policy.preferred_location==UVM_ID_CPU && (r.policy.accessed_by&(1u<<1)));
    uvm_va_block_t b={.gpu_prefetch_started=3,.gpu_prefetch_retry=3}; r.blocks=&b;
    assert(uvm_cpu_block_policy_should_add_accessed_by(&r,&g));
    assert(!(b.gpu_prefetch_started&(1u<<1)) && !(b.gpu_prefetch_retry&(1u<<1)));
    assert(uvm_cpu_block_policy_should_service_4k(&r,&g));
    r.cpu_access_counter_policy=false; assert(!uvm_cpu_block_policy_should_service_4k(&r,&g));
    uvm_va_range_managed_t optout={.va_range={&v},.policy={.preferred_location=7}};
    uvm_cpu_preferred_blocks_enable=0; uvm_cpu_block_policy_init_range(&optout);
    assert(!optout.cpu_access_counter_policy && optout.policy.preferred_location==7 && !optout.policy.accessed_by);
    uvm_cpu_preferred_blocks_enable=1; g_uvm_global.conf_computing_enabled=true;
    uvm_cpu_block_policy_init_range(&optout); assert(!optout.cpu_access_counter_policy);
    g_uvm_global.conf_computing_enabled=false;
    for (int kind=0;kind<3;++kind) {
        uvm_va_range_managed_t excluded={.va_range={&v}};
        p.is_integrated_gpu=kind==0; p.access_counters_supported=kind!=1;
        v.accessible_from[0]=kind==2?0:2;
        uvm_cpu_block_policy_init_range(&excluded);
        assert(excluded.cpu_access_counter_policy && !excluded.policy.accessed_by);
        assert(!uvm_cpu_block_policy_should_service_4k(&excluded,&g));
    }
    return 0;
}
'''
    compile_run(code, "legacy-policy")

def compile_run(code, name):
    with tempfile.TemporaryDirectory(prefix=name) as d:
        c = Path(d) / "test.c"; out = Path(d) / "test"; c.write_text(code)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", str(c), "-o", str(out)], check=True)
        subprocess.run([str(out)], check=True)

def run_promotion_harness():
    source = (ROOT / "kernel-open/nvidia-uvm/uvm_gpu_access_counters.c").read_text()
    block = extract_block(source, "if (promote_full_block)")
    code = r'''
#include <assert.h>
#include <stdbool.h>
typedef unsigned NvU32; typedef struct { unsigned bits; } uvm_page_mask_t;
#define UVM_ID_CPU 0
#define NUMA_NO_NODE (-1)
#define NV_OK 0
static uvm_page_mask_t cpu={.bits=0xf}, gpu={.bits=0x3};
static bool cpu_present=true, gpu_present=true;
static const uvm_page_mask_t *uvm_va_block_resident_mask_get(void*a,int b,int c){(void)a;(void)c;return b==UVM_ID_CPU?(cpu_present?&cpu:0):&gpu;}
static void uvm_page_mask_copy(uvm_page_mask_t*d,const uvm_page_mask_t*s){*d=*s;}
static void uvm_page_mask_andnot(uvm_page_mask_t*d,const uvm_page_mask_t*s,const uvm_page_mask_t*r){d->bits=s->bits & ~r->bits;}
static int run(uvm_page_mask_t *accessed, bool promote_full_block) {
    void *va_block = 0; uvm_page_mask_t *accessed_pages = accessed;
    const uvm_page_mask_t *residency_mask = gpu_present?&gpu:0;
'''+block+r'''
    return NV_OK;
}
int main(void) {
    uvm_page_mask_t accessed={.bits=2};
    assert(run(&accessed, true) == NV_OK && accessed.bits == (cpu.bits & ~gpu.bits));
    accessed.bits=2;
    assert(run(&accessed, false) == NV_OK && accessed.bits == 2);
    gpu_present=false; assert(run(&accessed,true)==NV_OK && accessed.bits==cpu.bits);
    cpu_present=false; accessed.bits=2; assert(run(&accessed,true)==NV_OK && accessed.bits==2);
    return 0;
}
'''
    compile_run(code, "legacy-promotion")

if __name__ == "__main__":
    run_policy_harness(); run_promotion_harness(); print("legacy policy mapping and full-block promotion checks passed")
