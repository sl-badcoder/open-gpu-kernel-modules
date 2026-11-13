"""Compile and exercise the bounded sparse remote co-demand learner."""
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


HARNESS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include "uvm_prefetch_pattern.h"
static void obs(uvm_prefetch_pattern_t *t, uint64_t r, unsigned n, uint64_t b) {
    bool c;
    assert(uvm_prefetch_pattern_observe(t, r, (uint64_t)n << 22, 0, b, &c));
}
static void pair(uvm_prefetch_pattern_t *t, unsigned a, unsigned z, uint64_t first) {
    for (uint64_t b=first; b<first+3; b++) { obs(t,7,a,b); obs(t,7,z,b); }
}
int main(void) {
    uvm_prefetch_pattern_t t = {0}, other = {0};
    bool c = true;
    assert(sizeof(t) < 48*1024);
    pair(&t,1,4,1);
    assert(uvm_prefetch_pattern_predict(&t,7,1ull<<22,0,3)==2);
    obs(&t,7,1,4);
    assert(uvm_prefetch_pattern_predict(&t,7,1ull<<22,0,4)==0x12);
    uint64_t events=t.events;
    assert(!uvm_prefetch_pattern_observe(&t,7,(1ull<<22)+(1ull<<21),0,4,&c));
    assert(!c && t.events==events);
    assert(uvm_prefetch_pattern_predict(&t,7,2ull<<22,0,4)==4);
    assert(uvm_prefetch_pattern_predict(&t,8,1ull<<22,0,4)==2);
    assert(uvm_prefetch_pattern_predict(&t,7,1ull<<22,1,4)==2);
    assert(uvm_prefetch_pattern_predict(&other,7,1ull<<22,0,4)==2);
    obs(&t,7,4,4);
    assert(uvm_prefetch_pattern_predict(&t,7,1ull<<22,0,4)==2);
    // A new repeatedly observed group replaces the old phase, never unions it.
    pair(&t,14,15,5); obs(&t,7,14,8);
    assert(uvm_prefetch_pattern_predict(&t,7,14ull<<22,0,8)==0xc000);
    assert(uvm_prefetch_pattern_find(&t,7,0,0)->learned==0xc000);
    // Reverse observation order learns the same complete-batch group.
    pair(&other,15,14,1); obs(&other,7,14,4);
    assert(uvm_prefetch_pattern_predict(&other,7,14ull<<22,0,4)==0xc000);
    for (unsigned i=0;i<100;i++)
        assert(uvm_prefetch_pattern_predict(&t,7,14ull<<22,0,8)==0xc000);
    // Advance event age without colliding with the target slot.
    uint64_t key=8;
    while(uvm_prefetch_pattern_hash(key,0,0)==uvm_prefetch_pattern_hash(7,0,0)) key++;
    for(uint64_t b=1;b<=512;b++) obs(&t,key,(unsigned)(b&15),b);
    assert(uvm_prefetch_pattern_predict(&t,7,14ull<<22,0,8)==0x4000);
    assert(t.expirations==1);
    // A direct-mapped replacement cannot inherit another allocation's history.
    key=8;
    while(uvm_prefetch_pattern_hash(key,0,0)!=uvm_prefetch_pattern_hash(7,0,0)) key++;
    obs(&other,key,14,1);
    assert(other.replacements==1);
    assert(uvm_prefetch_pattern_predict(&other,key,14ull<<22,0,1)==0x4000);
    assert(uvm_prefetch_pattern_predict(&other,7,14ull<<22,0,4)==0x4000);
    return 0;
}
'''


WRAPPER_PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#define UVM_ASSERT assert
#include "uvm_prefetch_budget.h"
#include "uvm_prefetch_pattern.h"
typedef struct {
    int list_lock;
    struct { uvm_prefetch_budget_t budget; uvm_prefetch_pattern_t patterns; } eager_eviction;
} uvm_pmm_gpu_t;
static bool uvm_perf_speculative_prefetch_sparse_enable = true;
static void uvm_spin_lock(int *lock) { assert(!*lock); *lock=1; }
static void uvm_spin_unlock(int *lock) { assert(*lock); *lock=0; }
#define NV_GETTIME() 1
static uvm_prefetch_budget_config_t speculative_prefetch_config(void) {
    uvm_prefetch_budget_config_t cfg={0};
    cfg.observation_ns=20; cfg.min_blocks=32;
    return cfg;
}
'''
WRAPPER_CASES = r'''
int main(void) {
    uvm_pmm_gpu_t p={0};
    for (unsigned b=1;b<=3;b++) {
        assert(uvm_pmm_gpu_prefetch_pattern(&p,7,1ull<<22,0,b,0x12)==2);
        assert(uvm_pmm_gpu_prefetch_pattern(&p,7,4ull<<22,0,b,0x12)==16);
    }
    assert(p.eager_eviction.budget.locality_events==6);
    assert(p.eager_eviction.budget.locality_confirmed==1);
    // Known peer demands must not be reported as learned suggestions.
    assert(uvm_pmm_gpu_prefetch_pattern(&p,7,1ull<<22,0,4,0x12)==2);
    assert(p.eager_eviction.patterns.learned_predictions==0);
    assert(uvm_pmm_gpu_prefetch_pattern(&p,7,1ull<<22,0,4,2)==0x12);
    assert(p.eager_eviction.patterns.learned_predictions==1);
    assert(p.eager_eviction.budget.locality_events==7);
    uvm_perf_speculative_prefetch_sparse_enable=false;
    assert(uvm_pmm_gpu_prefetch_pattern(&p,7,4ull<<22,0,4,0)==0);
    assert(p.eager_eviction.budget.locality_events==7 && !p.list_lock);
    return 0;
}
'''


def test_prefetch_pattern_compiles_and_runs():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "pattern.c"
        exe = Path(td) / "pattern"
        source = (ROOT / "kernel-open/nvidia-uvm/uvm_pmm_gpu.c").read_text()
        start = source.index("NvU16 uvm_pmm_gpu_prefetch_pattern(")
        end = source.index("\nNvU64 uvm_pmm_gpu_prefetch_record", start)
        wrapper = WRAPPER_PRELUDE + source[start:end] + WRAPPER_CASES
        for harness in (HARNESS, wrapper):
            src.write_text(harness)
            compile_and_run(src, exe)


def compile_and_run(src, exe):
    subprocess.run([
        "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
        "-I", str(ROOT / "kernel-open/common/inc"),
        "-I", str(ROOT / "kernel-open/nvidia-uvm"), str(src), "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    test_prefetch_pattern_compiles_and_runs()
