#!/usr/bin/env python3
"""Exercise production diagnostics with a deterministic clock and atomics."""
from pathlib import Path
import subprocess
import tempfile
from test_demand_first_eviction import function

ROOT = Path(__file__).parents[2]


def main():
    source = (ROOT / "kernel-open/nvidia-uvm/uvm_cpu_block_policy.c").read_text()
    header = (ROOT / "kernel-open/nvidia-uvm/uvm_perf_diagnostics.h").read_text()
    header = header.replace('#include "uvm_common.h"', '')
    code = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
typedef uint64_t NvU64;
typedef uint64_t atomic64_t;
static NvU64 now;
static unsigned clock_reads;
static NvU64 clock_read(void) { ++clock_reads; return now; }
#define NV_GETTIME() clock_read()
#define UVM_ASSERT(x) assert(x)
static void atomic64_inc(atomic64_t *v) { ++*v; }
static void atomic64_add(NvU64 n, atomic64_t *v) { *v += n; }
''' + header + r'''
bool uvm_perf_placement_diagnostics_enable;
static atomic64_t placement_diag_calls[UVM_PERF_DIAG_STAGE_COUNT];
static atomic64_t placement_diag_ns[UVM_PERF_DIAG_STAGE_COUNT];
static atomic64_t uvm_perf_diag_admitted_empty, uvm_perf_diag_admitted_nonempty;
static atomic64_t uvm_perf_diag_admitted_failed;
'''
    code += function(source, 'uvm_perf_diag_record')
    code += function(source, 'uvm_perf_diag_population')
    code += r'''
int main(void) {
    now=100;
    assert(!uvm_perf_diag_begin() && !clock_reads);
    uvm_perf_diag_end(UVM_PERF_DIAG_COUNTER_SERVICE,0);
    uvm_perf_diag_population(100,false);
    assert(!clock_reads && !uvm_perf_diag_admitted_nonempty && !uvm_perf_diag_admitted_failed);
    uvm_perf_placement_diagnostics_enable=true;
    NvU64 start=uvm_perf_diag_begin();now=175;
    uvm_perf_diag_end(UVM_PERF_DIAG_COUNTER_SERVICE,start);
    assert(clock_reads==2 && placement_diag_calls[UVM_PERF_DIAG_COUNTER_SERVICE]==1);
    assert(placement_diag_ns[UVM_PERF_DIAG_COUNTER_SERVICE]==75);
    start=uvm_perf_diag_begin();now=225;
    uvm_perf_diag_end(UVM_PERF_DIAG_COUNTER_SERVICE,start);
    assert(placement_diag_calls[UVM_PERF_DIAG_COUNTER_SERVICE]==2);
    assert(placement_diag_ns[UVM_PERF_DIAG_COUNTER_SERVICE]==125);
    for (unsigned i=1;i<UVM_PERF_DIAG_STAGE_COUNT;i++) {
        start=uvm_perf_diag_begin();now+=10;
        uvm_perf_diag_end((uvm_perf_diag_stage_t)i,start);
        assert(placement_diag_calls[i]==1 && placement_diag_ns[i]==10);
    }
    uvm_perf_diag_population(0,true);
    uvm_perf_diag_population(4096,true);
    uvm_perf_diag_population(0,false);
    uvm_perf_diag_population(4096,false);
    assert(uvm_perf_diag_admitted_empty==2 && uvm_perf_diag_admitted_nonempty==2);
    assert(uvm_perf_diag_admitted_failed==2);
    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="placement-diagnostics-") as directory:
        cfile = Path(directory) / "test.c"
        binary = Path(directory) / "test"
        cfile.write_text(code)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", str(cfile), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
    print("Placement diagnostics: disabled hooks, stage isolation, durations and outcomes passed.")


if __name__ == "__main__":
    main()
