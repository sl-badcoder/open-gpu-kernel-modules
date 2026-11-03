#!/usr/bin/env python3
import pathlib
import subprocess
import tempfile


def main():
    root = pathlib.Path(__file__).parents[2]
    header = (root / "kernel-open/nvidia-uvm/uvm_vab.h").read_text()
    header = header.replace('#include "nvtypes.h"', "")
    source = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
typedef uint64_t NvU64;
typedef uint32_t NvU32;
''' + header + r'''
static uvm_vab_snapshot_t s;
static void setup(unsigned g) {
    s = (uvm_vab_snapshot_t){0}; s.valid = true;
    s.linear_base = 0x10000000; s.sampled_ns = 100; s.granularity = g;
    for (unsigned i = 0; i < UVM_VAB_WORD_COUNT; ++i) s.enabled_mask[i] = ~0ULL;
}
int main(void) {
    setup(6); /* 4 MiB: both 2 MiB roots share one native bit. */
    assert(uvm_vab_classify(&s, s.linear_base, 2ULL << 20, 100) == UVM_VAB_REGION_COLD);
    s.access_bits[0] = 1;
    assert(uvm_vab_classify(&s, s.linear_base + (2ULL << 20), 2ULL << 20, 100) == UVM_VAB_REGION_HOT);
    setup(0); /* 64 KiB native granularity. */
    s.access_bits[0] = 1ULL << 1;
    assert(uvm_vab_classify(&s, s.linear_base + (64ULL << 10), 64ULL << 10, 100) == UVM_VAB_REGION_HOT);
    assert(uvm_vab_classify(&s, s.linear_base, 64ULL << 10, 100) == UVM_VAB_REGION_COLD);
    s.enabled_mask[0] &= ~1ULL;
    assert(uvm_vab_classify(&s, s.linear_base, 64ULL << 10, 100) == UVM_VAB_REGION_UNKNOWN);
    setup(0);
    assert(uvm_vab_classify(&s, s.linear_base + (4096ULL << 16), 1, 100) == UVM_VAB_REGION_UNKNOWN);
    assert(uvm_vab_classify(&s, ~0ULL, 2, 100) == UVM_VAB_REGION_UNKNOWN);
    s.linear_base = ~0ULL;
    assert(uvm_vab_classify(&s, 0, 1, 100) == UVM_VAB_REGION_UNKNOWN);
    setup(0); s.sampled_ns = 100;
    assert(uvm_vab_classify(&s, s.linear_base, 1, 101) == UVM_VAB_REGION_UNKNOWN);
    assert(uvm_vab_sample_fresh(150, 100, 50));
    assert(!uvm_vab_sample_fresh(151, 100, 50));
    assert(!uvm_vab_sample_fresh(99, 100, 50));
    assert(!uvm_vab_sample_fresh(100, 0, 50));
    return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
        c = pathlib.Path(directory) / "test.c"
        exe = pathlib.Path(directory) / "test"
        c.write_text(source)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", str(c), "-o", str(exe)],
                       check=True)
        subprocess.run([str(exe)], check=True)
    print("VAB classifier tests passed")


if __name__ == "__main__":
    main()
