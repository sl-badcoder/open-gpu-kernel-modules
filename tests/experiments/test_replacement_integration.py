#!/usr/bin/env python3
"""Exercise the production replacement-accounting hooks with small masks."""
from pathlib import Path
import subprocess
import tempfile
from test_demand_first_eviction import function

ROOT = Path(__file__).parents[2]

PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef uint64_t NvU64; typedef unsigned NvU32;
typedef NvU64 uvm_page_mask_t;
#define PAGE_SIZE 4096ULL
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
typedef struct { bool cpu_access_counter_policy; } managed_range_t;
typedef struct {
    NvU64 replacement_eviction_stamp;
    uvm_page_mask_t replacement_live, replacement_evicted;
} uvm_va_block_gpu_state_t;
typedef struct {
    int lock; managed_range_t *managed_range; uvm_va_block_gpu_state_t state;
} uvm_va_block_t;
typedef struct { struct { uvm_page_mask_t pages_changed_residency; } make_resident; } uvm_va_block_context_t;
static bool turnover_enabled = true;
static bool uvm_pmm_gpu_prefetch_turnover_enabled(void) { return turnover_enabled; }
static void uvm_assert_mutex_locked(int *lock) { assert(*lock); }
static uvm_va_block_gpu_state_t *uvm_va_block_gpu_state_get(uvm_va_block_t *b, int id)
{ (void)id; assert(b->lock); return &b->state; }
static void uvm_spin_lock(int *lock) { assert(!*lock); *lock = 1; }
static void uvm_spin_unlock(int *lock) { assert(*lock); *lock = 0; }
static bool uvm_page_mask_empty(const uvm_page_mask_t *m) { return *m == 0; }
static unsigned uvm_page_mask_weight(const uvm_page_mask_t *m) { return __builtin_popcountll(*m); }
static void uvm_page_mask_and(uvm_page_mask_t *d, const uvm_page_mask_t *a, const uvm_page_mask_t *b) { *d = *a & *b; }
static void uvm_page_mask_or(uvm_page_mask_t *d, const uvm_page_mask_t *a, const uvm_page_mask_t *b) { *d = *a | *b; }
static void uvm_page_mask_andnot(uvm_page_mask_t *d, const uvm_page_mask_t *a, const uvm_page_mask_t *b) { *d = *a & ~*b; }
static void uvm_page_mask_zero(uvm_page_mask_t *d) { *d = 0; }
'''

POSTLUDE = r'''
typedef struct { uvm_prefetch_replacement_state_t replacement; int list_lock; } eager_t;
typedef struct { eager_t eager_eviction; int list_lock; } uvm_pmm_gpu_t;
typedef struct { int id; uvm_pmm_gpu_t pmm; } uvm_gpu_t;
'''

CASES = r'''
static uvm_gpu_t gpu;
static uvm_va_block_t block;
static managed_range_t managed;
static uvm_va_block_context_t context;

static void reset(void)
{
    memset(&gpu, 0, sizeof(gpu)); memset(&block, 0, sizeof(block));
    memset(&managed, 0, sizeof(managed)); memset(&context, 0, sizeof(context));
    block.lock = 1; block.managed_range = &managed;
    managed.cpu_access_counter_policy = true;
    turnover_enabled = true;
}

int main(void)
{
    reset();
    // Four pages arrive after two pages were evicted. Only the intersection
    // contributes to repopulation, and the live gauge does not double count.
    block.state.replacement_evicted = 0x3;
    block.state.replacement_live = 0x4;
    gpu.pmm.eager_eviction.replacement.resident_bytes = PAGE_SIZE;
    context.make_resident.pages_changed_residency = 0xf;
    uvm_cpu_block_policy_record_population(&block, &gpu, &context, false);
    assert(gpu.pmm.eager_eviction.replacement.population_bytes == 4 * PAGE_SIZE);
    assert(gpu.pmm.eager_eviction.replacement.repopulated_bytes == 2 * PAGE_SIZE);
    assert(gpu.pmm.eager_eviction.replacement.resident_bytes == 4 * PAGE_SIZE);
    assert(block.state.replacement_live == 0xf && block.state.replacement_evicted == 0);
    assert(gpu.pmm.eager_eviction.replacement.window_population_bytes == 4 * PAGE_SIZE);

    // A repeated population of already-live pages does not increment the
    // resident gauge a second time.
    context.make_resident.pages_changed_residency = 0x4;
    uvm_cpu_block_policy_record_population(&block, &gpu, &context, false);
    assert(gpu.pmm.eager_eviction.replacement.resident_bytes == 4 * PAGE_SIZE);

    // Partial eviction stamps once, tracks only live pages, and preserves the
    // first stamp across a second partial eviction.
    uvm_cpu_block_policy_record_eviction(&block, &gpu, &(uvm_page_mask_t){0x5});
    assert(gpu.pmm.eager_eviction.replacement.evicted_bytes == 2 * PAGE_SIZE);
    assert(gpu.pmm.eager_eviction.replacement.resident_bytes == 2 * PAGE_SIZE);
    assert(block.state.replacement_live == 0xa && block.state.replacement_evicted == 0x5);
    NvU64 first_stamp = block.state.replacement_eviction_stamp;
    uvm_cpu_block_policy_record_eviction(&block, &gpu, &(uvm_page_mask_t){0x8});
    assert(gpu.pmm.eager_eviction.replacement.evicted_bytes == 3 * PAGE_SIZE);
    assert(gpu.pmm.eager_eviction.replacement.resident_bytes == PAGE_SIZE);
    assert(block.state.replacement_eviction_stamp == first_stamp);

    // Native/untracked eviction counts bytes removed, but only tracked live
    // pages are used for the resident gauge.
    uvm_cpu_block_policy_record_eviction(&block, &gpu, &(uvm_page_mask_t){0x1});
    assert(gpu.pmm.eager_eviction.replacement.evicted_bytes == 4 * PAGE_SIZE);
    assert(gpu.pmm.eager_eviction.replacement.resident_bytes == PAGE_SIZE);

    // A neighbor updates global population but cannot feed the trigger
    // controller's window.
    reset(); block.state.replacement_live = 0;
    gpu.pmm.eager_eviction.replacement.resident_bytes = 0;
    context.make_resident.pages_changed_residency = 0x3;
    uvm_cpu_block_policy_record_population(&block, &gpu, &context, true);
    assert(gpu.pmm.eager_eviction.replacement.population_bytes == 2 * PAGE_SIZE);
    assert(gpu.pmm.eager_eviction.replacement.window_population_bytes == 0);

    // Forgetting a block removes its live bytes exactly once.
    uvm_cpu_block_policy_forget_population(&block, &gpu);
    assert(gpu.pmm.eager_eviction.replacement.resident_bytes == 0);
    assert(block.state.replacement_live == 0);
    uvm_cpu_block_policy_forget_population(&block, &gpu);
    assert(gpu.pmm.eager_eviction.replacement.resident_bytes == 0);

    // Disabled accounting is a complete no-op.
    reset(); turnover_enabled = false; context.make_resident.pages_changed_residency = 0xf;
    uvm_cpu_block_policy_record_population(&block, &gpu, &context, false);
    uvm_cpu_block_policy_record_eviction(&block, &gpu, &(uvm_page_mask_t){0x3});
    assert(gpu.pmm.eager_eviction.replacement.population_bytes == 0);
    assert(gpu.pmm.eager_eviction.replacement.evicted_bytes == 0);
    puts("Replacement accounting hooks: partial return, live tracking, neighbor exclusion, forget and disable passed.");
    return 0;
}
'''

def main():
    source = (ROOT / 'kernel-open/nvidia-uvm/uvm_cpu_block_policy.c').read_text()
    replacement = (ROOT / 'kernel-open/nvidia-uvm/uvm_prefetch_replacement.h').read_text()
    code = PRELUDE + replacement + POSTLUDE + function(source, 'uvm_cpu_block_policy_record_population')
    code += function(source, 'uvm_cpu_block_policy_record_eviction')
    code += function(source, 'uvm_cpu_block_policy_forget_population') + CASES
    with tempfile.TemporaryDirectory(prefix='replacement-integration-') as directory:
        c = Path(directory) / 'test.c'; binary = Path(directory) / 'test'
        c.write_text(code)
        subprocess.run(['cc', '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                        '-fsanitize=address,undefined', '-g', str(c), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

if __name__ == '__main__':
    main()
