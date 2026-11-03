#!/usr/bin/env python3
"""Validate conservative physical-window access-bit interpretation."""

from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).parents[2]


def main():
    source = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
typedef uint64_t NvU64; typedef uint32_t NvU32;
''' + (ROOT / "kernel-open/nvidia-uvm/uvm_vab.h").read_text().replace('#include "nvtypes.h"', '') + r'''
int main(void) {
    NvU64 bits[2] = {0}, mask[2] = {~0ULL, ~0ULL};
    assert(uvm_vab_region_cold(bits, mask, 2, 0, 0, 0, 2 * 1024 * 1024));
    mask[0] &= ~(1ULL << 7);
    assert(!uvm_vab_region_cold(bits, mask, 2, 0, 0, 0, 2 * 1024 * 1024));
    mask[0] = ~0ULL;
    assert(!uvm_vab_region_cold(bits, mask, 2, 0, 0, 127 * 65536, 2 * 65536));
    assert(!uvm_vab_region_cold(bits, mask, 2, 0, 0, UINT64_MAX, 2));
    assert(!uvm_vab_region_cold(bits, mask, 2, 16, 0, 0, 1));
    // Granularity 0 is 64 KiB. A root spanning two bitmap entries is hot if
    // either entry reports an access.
    bits[0] = 1ULL << 2;
    assert(!uvm_vab_region_cold(bits, mask, 2, 0, 0, 0, 2 * 1024 * 1024));
    assert(uvm_vab_region_accessed(bits, mask, 2, 0, 0, 2 * 65536, 2 * 65536));
    bits[0] = 0; bits[1] = 1ULL << 0;
    assert(uvm_vab_region_accessed(bits, mask, 2, 0, 0, 64 * 65536, 64 * 65536));
    // Disabled bits and untouched regions provide no positive evidence.
    mask[1] = 0; assert(!uvm_vab_region_accessed(bits, mask, 2, 0, 0, 64 * 65536, 65536));
    // Coarser RM granularity maps the same root to one coarse bit.
    bits[0] = 1; mask[1] = ~0ULL;
    assert(uvm_vab_region_accessed(bits, mask, 2, 5, 0, 0, 2 * 1024 * 1024));
    assert(!uvm_vab_region_accessed(bits, mask, 2, 5, 0, 2 * 1024 * 1024, 2 * 1024 * 1024));
    // Invalid/unmapped intervals and stale samples supply no positive evidence.
    assert(!uvm_vab_region_accessed(bits, mask, 2, 0, 65536, 0, 65536));
    assert(!uvm_vab_region_accessed(bits, mask, 2, 16, 0, 0, 1));
    assert(!uvm_vab_region_accessed(bits, mask, 2, 0, 0, UINT64_MAX, 2));
    assert(uvm_vab_sample_fresh(100, 90, 10));
    assert(!uvm_vab_sample_fresh(101, 90, 10));
    assert(!uvm_vab_sample_fresh(90, 100, 10));
    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="uvm-vab-") as directory:
        c = Path(directory) / "test.c"
        binary = Path(directory) / "test"
        c.write_text(source)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", str(c), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

    # Extract the production used-list selector. This keeps the test coupled
    # to the actual PMM implementation while replacing only kernel list and
    # GPU accessors with a tiny deterministic model.
    pmm_source = (ROOT / "kernel-open/nvidia-uvm/uvm_pmm_gpu.c").read_text()
    start = pmm_source.index("static uvm_gpu_chunk_t *get_first_allocated_chunk")
    end = pmm_source.index("static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict", start)
    selector = pmm_source[start:end]
    vab_header = (ROOT / "kernel-open/nvidia-uvm/uvm_vab.h").read_text().replace('#include "nvtypes.h"', '')
    harness = r'''
#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
typedef uint64_t NvU64; typedef unsigned NvU32;
typedef struct list_head { struct list_head *next, *prev; } list_head;
typedef struct uvm_gpu_chunk_struct { list_head list; NvU64 address; } uvm_gpu_chunk_t;
typedef uvm_gpu_chunk_t uvm_gpu_root_chunk_t;
typedef enum { UVM_PMM_ALLOC_LIST_UNUSED, UVM_PMM_ALLOC_LIST_DISCARDED, UVM_PMM_ALLOC_LIST_USED, UVM_PMM_ALLOC_LIST_COUNT } uvm_pmm_alloc_list_t;
typedef struct { list_head alloc_list[UVM_PMM_ALLOC_LIST_COUNT]; } roots_t;
typedef struct { roots_t root_chunks; } uvm_pmm_gpu_t;
typedef struct { int id; } uvm_parent_gpu_t;
typedef struct { uvm_parent_gpu_t *parent; } uvm_gpu_t;
typedef struct { NvU64 currentBits[2], enabledMask[2]; NvU32 granularity; } UvmGpuAccessBitsBufferAlloc;
static uvm_gpu_t gpu;
#define container_of(p,t,m) ((t *)((char *)(p) - offsetof(t,m)))
#define list_first_chunk(h) ((h)->next == (h) ? NULL : container_of((h)->next, uvm_gpu_chunk_t, list))
#define list_for_each_entry(pos, h, member) for (pos = list_first_chunk(h); pos; pos = ((pos)->list.next == (h) ? NULL : container_of((pos)->list.next, uvm_gpu_chunk_t, member)))
#define uvm_assert_spinlock_locked(x) ((void)0)
#define uvm_pmm_to_gpu(x) (&gpu)
#define UVM_CHUNK_SIZE_MAX (2ULL * 1024 * 1024)
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
''' + vab_header + r'''
static UvmGpuAccessBitsBufferAlloc snapshot; static NvU64 snapshot_base; static bool snapshot_valid; static unsigned snapshots;
static bool uvm_gpu_vab_snapshot(uvm_parent_gpu_t *p, UvmGpuAccessBitsBufferAlloc *out, NvU64 *base) { (void)p; ++snapshots; if (!snapshot_valid) return false; *out=snapshot; *base=snapshot_base; return true; }
''' + selector + r'''
static void init(list_head *h) { h->next = h->prev = h; }
static void add(list_head *h, uvm_gpu_chunk_t *c) { c->list.next=h; c->list.prev=h->prev; h->prev->next=&c->list; h->prev=&c->list; }
int main(void) {
    assert(uvm_vab_sample_fresh(100, 90, 10));
    uvm_pmm_gpu_t p={0}; for (int i=0;i<UVM_PMM_ALLOC_LIST_COUNT;i++) init(&p.root_chunks.alloc_list[i]);
    uvm_gpu_chunk_t unused={0}, discarded={0}, hot={.address=0}, cold={.address=UVM_CHUNK_SIZE_MAX};
    add(&p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_UNUSED], &unused);
    add(&p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_DISCARDED], &discarded);
    add(&p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_USED], &hot); add(&p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_USED], &cold);
    snapshot.granularity=0; snapshot.enabledMask[0]=~0ULL; snapshot.currentBits[0]=1;
    snapshot_base=0; snapshot_valid=true; assert(get_first_allocated_chunk(&p)==&unused);
    p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_UNUSED].next=&p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_UNUSED];
    assert(get_first_allocated_chunk(&p)==&discarded && snapshots==0);
    p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_DISCARDED].next=&p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_DISCARDED];
    assert(get_first_allocated_chunk(&p)==&cold && snapshots==1);
    snapshot.currentBits[0]=1ULL << 32; assert(get_first_allocated_chunk(&p)==&hot && snapshots==2);
    // A root with missing enabled bits is unknown, not cold. Skip it in
    // favor of a fully tracked zero root even when both raw regions are zero.
    snapshot.currentBits[0]=0; snapshot.enabledMask[0]=~1ULL;
    assert(get_first_allocated_chunk(&p)==&cold);
    p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_USED].next=&p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_USED];
    assert(get_first_allocated_chunk(&p)==NULL);
    // Coarse 2 MiB bits: all roots accessed falls back to the oldest root.
    uvm_gpu_chunk_t first={.address=0}, second={.address=2ULL * 1024 * 1024};
    init(&p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_USED]);
    add(&p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_USED], &first);
    add(&p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_USED], &second);
    snapshot.granularity=5; snapshot.currentBits[0]=3; snapshot.enabledMask[0]=3;
    snapshot_valid=true; assert(get_first_allocated_chunk(&p)==&first);
    // A skipped/failed sample disables positive hints and also falls back.
    snapshot_valid=false; assert(get_first_allocated_chunk(&p)==&first);
}
'''
    with tempfile.TemporaryDirectory(prefix="uvm-vab-selector-") as directory:
        c = Path(directory) / "selector.c"
        binary = Path(directory) / "selector"
        c.write_text(harness)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", str(c), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
