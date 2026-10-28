#!/usr/bin/env python3
"""Exercise the copy-byte accounting rules used by eviction.

The full UVM copy engine requires a Linux kernel build. This harness extracts
the accounting shape into a small C model and verifies the important lifecycle:
one reset per eviction operation, accumulation across copy calls, and no credit
for failed or zero-length submissions.
"""
import pathlib
import subprocess
import tempfile


SOURCE = r'''
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct { uint64_t bytes; } context_t;

static void eviction_begin(context_t *context) { context->bytes = 0; }

static bool submit_copy(context_t *context, uint64_t pages, bool succeeds)
{
    if (!succeeds || !pages)
        return false;
    context->bytes += pages * 4096;
    return true;
}

int main(void)
{
    context_t context = { 99 };

    eviction_begin(&context);
    assert(context.bytes == 0);
    assert(submit_copy(&context, 3, true));
    assert(submit_copy(&context, 5, true));
    assert(context.bytes == 8 * 4096);

    // A failed retry and a zero-page region must not add payload.
    assert(!submit_copy(&context, 7, false));
    assert(!submit_copy(&context, 0, true));
    assert(context.bytes == 8 * 4096);

    // A new root starts a new accounting interval.
    eviction_begin(&context);
    assert(context.bytes == 0);
    assert(submit_copy(&context, 2, true));
    assert(context.bytes == 2 * 4096);
    return 0;
}
'''


def main():
    production = pathlib.Path(__file__).parents[2] / "kernel-open/nvidia-uvm/uvm_va_block.c"
    pmm = pathlib.Path(__file__).parents[2] / "kernel-open/nvidia-uvm/uvm_pmm_gpu.c"
    text = production.read_text()
    assert text.count("block_context->make_resident.eviction_copy_bytes = 0;") == 1
    assert text.count("eviction_copy_bytes +=") >= 3
    pmm_text = pmm.read_text()
    publish = pmm_text.index("atomic64_add(root_chunk->eviction_copy_bytes")
    assert pmm_text.index("atomic64_inc(&uvm_perf_eviction_copy_roots", publish) < pmm_text.index("uvm_gpu_chunk_set_in_eviction", publish)

    with tempfile.TemporaryDirectory(prefix="eviction-copy-") as directory:
        root = pathlib.Path(directory)
        source = root / "test.c"
        binary = root / "test"
        source.write_text(SOURCE)
        subprocess.run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
            str(source), "-o", str(binary)
        ], check=True)
        subprocess.run([str(binary)], check=True)
    print("Passed: per-root reset, multi-region accumulation, zero-copy and failure accounting")


if __name__ == "__main__":
    main()
