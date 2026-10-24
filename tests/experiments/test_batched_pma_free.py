#!/usr/bin/env python3
"""Compile production PMA-free functions against a mocked kernel API."""

import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).parents[2]
SOURCE = ROOT / "kernel-open/nvidia-uvm/uvm_pmm_gpu.c"


def extract_definition(source, signature):
    """Extract a definition, skipping declarations with a semicolon."""
    offset = 0
    while True:
        start = source.find(signature, offset)
        assert start >= 0, signature
        brace = source.find("{", start + len(signature))
        if source.find(";", start + len(signature), brace) < 0:
            depth = 0
            for position in range(brace, len(source)):
                if source[position] == "{":
                    depth += 1
                elif source[position] == "}":
                    depth -= 1
                    if depth == 0:
                        return source[start:position + 1]
        offset = brace + 1


def main():
    source = SOURCE.read_text()
    helper = extract_definition(source, "static void free_root_chunks(uvm_pmm_gpu_t *pmm,")
    wrapper = extract_definition(source, "static void free_root_chunk(uvm_pmm_gpu_t *pmm,")
    eager_start = source.index("static unsigned eager_eviction_batch(")
    eager_body = source[eager_start:source.index("static void eager_eviction_work", eager_start)]
    assert "free_root_chunks(pmm, pending, count" in eager_body

    c = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
typedef uint64_t NvU64; typedef uint32_t NvU32; typedef int NV_STATUS;
enum { NV_OK = 0, NV_ERR_GENERIC = 1 };
enum { UVM_PMA_CALLED_FROM_PMA_EVICTION = 1, UVM_PMA_FREE_IS_ZERO = 2 };
enum { UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED = 7, UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED = 8 };
typedef enum { FREE_ROOT_CHUNK_MODE_DEFAULT, FREE_ROOT_CHUNK_MODE_PMA_EVICTION, FREE_ROOT_CHUNK_MODE_SKIP_PMA_FREE } free_root_chunk_mode_t;
#define UVM_CHUNK_SIZE_MAX 0x200000
struct tracker { int unused; };
struct chunk { NvU64 address; int state; int is_zero; int list; };
struct root { struct tracker tracker; struct chunk chunk; int waited; };
struct pmm { int pma_lock; int *pma; int list_lock; };
struct gpu { int unused; };
typedef struct pmm uvm_pmm_gpu_t; typedef struct gpu uvm_gpu_t;
typedef struct root uvm_gpu_root_chunk_t; typedef struct chunk uvm_gpu_chunk_t;
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define UVM_ASSERT(x) assert(x)
#define UVM_ASSERT_MSG(x, ...) do { assert(x); if (!(x)) fprintf(stderr, __VA_ARGS__); } while (0)
#define list_empty(x) (1)
static int read_locked, lock_depth, free_calls, free_count, wait_calls, wait_failure;
static NvU32 last_flags; static NvU64 freed[16]; static int global_status;
static struct root *observed_roots[16]; static unsigned observed_root_count; static struct root *active_root;
static uvm_gpu_t global_gpu;
static uvm_gpu_t *uvm_pmm_to_gpu(uvm_pmm_gpu_t *pmm) { (void)pmm; return &global_gpu; }
static const char *uvm_gpu_name(uvm_gpu_t *gpu) { (void)gpu; return "mock"; }
static const char *uvm_pmm_gpu_chunk_state_string(int state) { (void)state; return "mock"; }
static void uvm_down_read(int *lock) { (void)lock; assert(!read_locked); read_locked = 1; }
static void uvm_up_read(int *lock) { (void)lock; assert(read_locked); read_locked = 0; }
static void root_chunk_lock(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root) { (void)pmm; active_root = root; ++lock_depth; }
static void root_chunk_unlock(uvm_pmm_gpu_t *pmm, uvm_gpu_root_chunk_t *root) { (void)pmm; (void)root; assert(lock_depth); --lock_depth; }
static NV_STATUS uvm_tracker_wait_deinit(struct tracker *tracker) { (void)tracker; assert(read_locked); active_root->waited = 1; ++wait_calls; return wait_failure ? NV_ERR_GENERIC : NV_OK; }
static int uvm_global_get_status(void) { return global_status; }
static void uvm_spin_lock(int *lock) { (void)lock; assert(read_locked); }
static void uvm_spin_unlock(int *lock) { (void)lock; assert(read_locked); }
static void chunk_unpin(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, int state) { (void)pmm; assert(read_locked); chunk->state = state; }
static void nvUvmInterfacePmaFreePages(int *pma, NvU64 *addresses, unsigned count, NvU64 page_size, NvU32 flags) {
    (void)pma; assert(read_locked && page_size == 0x200000); ++free_calls; free_count = count; last_flags = flags;
    assert(count == observed_root_count);
    for (unsigned i = 0; i < count; ++i) {
        assert(addresses[i] == observed_roots[i]->chunk.address);
        assert(observed_roots[i]->waited);
        assert(observed_roots[i]->chunk.state == UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED);
        freed[i] = addresses[i];
    }
}
'''
    c += helper + "\n" + wrapper
    c += r'''
static void reset(void) { read_locked = lock_depth = free_calls = free_count = wait_calls = wait_failure = 0; last_flags = 0; global_status = NV_OK; }
int main(void) {
    struct pmm p = { 0 }; struct root storage[16]; struct root *roots[16];
    for (unsigned i = 0; i < 16; ++i) { storage[i].chunk.address = 0x1000 + i * 0x200000; storage[i].chunk.state = UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED; storage[i].chunk.is_zero = 0; roots[i] = &storage[i]; }
    observed_root_count = 0;
    free_root_chunks(&p, roots, 0, FREE_ROOT_CHUNK_MODE_DEFAULT);
    assert(!read_locked && !free_calls && !wait_calls);
    observed_root_count = 16;
    for (unsigned i = 0; i < 16; ++i) observed_roots[i] = roots[i];
    free_root_chunks(&p, roots, 16, FREE_ROOT_CHUNK_MODE_DEFAULT);
    assert(!read_locked && free_calls == 1 && free_count == 16 && wait_calls == 16 && freed[15] == storage[15].chunk.address);
    reset(); storage[0].chunk.state = UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED; storage[0].chunk.is_zero = 1;
    observed_root_count = 1; observed_roots[0] = roots[0];
    free_root_chunk(&p, roots[0], FREE_ROOT_CHUNK_MODE_DEFAULT);
    assert(free_calls == 1 && free_count == 1 && last_flags == UVM_PMA_FREE_IS_ZERO);
    reset(); storage[0].chunk.state = UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED; storage[0].chunk.is_zero = 0;
    observed_root_count = 1; observed_roots[0] = roots[0];
    free_root_chunk(&p, roots[0], FREE_ROOT_CHUNK_MODE_PMA_EVICTION);
    assert(free_calls == 1 && last_flags == UVM_PMA_CALLED_FROM_PMA_EVICTION);
    reset(); storage[0].chunk.state = UVM_PMM_GPU_CHUNK_STATE_TEMP_PINNED; wait_failure = 1; global_status = NV_ERR_GENERIC;
    observed_root_count = 1; observed_roots[0] = roots[0];
    free_root_chunk(&p, roots[0], FREE_ROOT_CHUNK_MODE_DEFAULT);
    assert(wait_calls == 1 && free_calls == 1 && !read_locked);
    puts("production batched PMA free harness passed"); return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
        c_path = pathlib.Path(directory) / "batched_pma_free.c"
        binary = pathlib.Path(directory) / "batched_pma_free"
        c_path.write_text(c)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", "-o", str(binary), str(c_path)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
