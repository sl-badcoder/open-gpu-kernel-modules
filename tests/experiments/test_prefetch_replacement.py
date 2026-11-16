#!/usr/bin/env python3
"""Standalone sanitizer tests for the turnover admission helper."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).parents[2]
HEADER = ROOT / "kernel-open/nvidia-uvm/uvm_prefetch_replacement.h"
PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
typedef uint32_t NvU32;
typedef uint64_t NvU64;
'''
CASES = r'''
static void low_churn_is_permissive(void) {
    uvm_prefetch_replacement_state_t s; uvm_prefetch_replacement_init(&s);
    for (int i = 0; i < 16; ++i)
        uvm_prefetch_replacement_record_population(&s, 4ULL<<20, 1, 4ULL<<30, 1ULL<<30, true);
    assert(s.evaluations == 1 && s.request_stride == 1);
    assert(s.last_population_bytes == 64ULL<<20 && s.last_short_quarter_bytes == 0);
}
static void high_churn_throttles_and_recovers(void) {
    uvm_prefetch_replacement_state_t s; uvm_prefetch_replacement_init(&s);
    for (int i = 0; i < 16; ++i)
        uvm_prefetch_replacement_record_population(&s, 4ULL<<20, 4ULL<<20, 1, 1ULL<<30, true);
    assert(s.request_stride == 4);
    for (int i = 0; i < 16; ++i)
        uvm_prefetch_replacement_record_population(&s, 4ULL<<20, 1, 1ULL<<30, 1ULL<<30, true);
    assert(s.request_stride == 1);
}
static void neighbors_do_not_feed_controller(void) {
    uvm_prefetch_replacement_state_t s; uvm_prefetch_replacement_init(&s);
    for (int i = 0; i < 64; ++i)
        uvm_prefetch_replacement_record_population(&s, 64ULL<<20, 64ULL<<20, 0, 1ULL<<30, false);
    assert(s.population_bytes == 4ULL<<30 && s.resident_bytes == 4ULL<<30 && s.trigger_population_bytes == 0 && s.evaluations == 0 && s.request_stride == 0);
}
static void boundaries_zero_budget_and_saturation(void) {
    uvm_prefetch_replacement_state_t s; uvm_prefetch_replacement_init(&s);
    uvm_prefetch_replacement_record_population(&s, 1, 1, 0, 0, true);
    assert(s.request_stride == 4);
    s.population_bytes = ~0ULL - 1;
    uvm_prefetch_replacement_record_population(&s, 100, 0, 0, 1ULL<<30, false);
    assert(s.population_bytes == ~0ULL);
    uvm_prefetch_replacement_record_population(&s, 1, 1, 1ULL<<28, 1ULL<<30, true);
    assert(s.short_quarter_bytes == 1);
    uvm_prefetch_replacement_record_eviction(&s, 10, 2);
    assert(s.evicted_bytes == 10 && s.resident_bytes == 100);
}
static void epoch_quota_is_bounded_and_refreshes(void) {
    uvm_prefetch_replacement_state_t s; uvm_prefetch_replacement_init(&s);
    s.request_stride = 4; s.resident_bytes = 64ULL << 20;
    assert(uvm_prefetch_replacement_limit(&s, 1, 100, 1ULL << 20) == 1);
    uvm_prefetch_replacement_consume(&s, 1);
    assert(uvm_prefetch_replacement_limit(&s, 1, 100, 1ULL << 20) == 0);
    assert(uvm_prefetch_replacement_limit(&s, 2, 100, 1ULL << 20) == 1);
    uvm_prefetch_replacement_consume(&s, 1);
    assert(uvm_prefetch_replacement_limit(&s, 0, 100, 1ULL << 20) == 0);
    assert(s.epoch_pauses == 2);
    s.request_stride = 1;
    assert(uvm_prefetch_replacement_limit(&s, 0, 100, 1ULL << 20) == 100);
}
static void epoch_budget_is_frozen_then_resized(void) {
    uvm_prefetch_replacement_state_t s = {.request_stride=4, .resident_bytes=128ULL<<20};
    assert(uvm_prefetch_replacement_limit(&s, 10, 100, 2ULL<<20)==1);
    uvm_prefetch_replacement_consume(&s, 1);
    s.resident_bytes=1024ULL<<20;
    assert(uvm_prefetch_replacement_limit(&s, 10, 100, 2ULL<<20)==0);
    assert(uvm_prefetch_replacement_limit(&s, 11, 100, 2ULL<<20)==8);
    uvm_prefetch_replacement_consume(&s, 3);
    assert(uvm_prefetch_replacement_limit(&s, 11, 100, 2ULL<<20)==5);
    assert(uvm_prefetch_replacement_limit(&s, 10, 100, 2ULL<<20)==0);
    s.resident_bytes=0;
    assert(uvm_prefetch_replacement_limit(&s, 12, 100, 2ULL<<20)==1);
}
static void ratio_hysteresis_and_distance_boundaries(void) {
    uvm_prefetch_replacement_state_t s = {0};
    const NvU64 batch=64ULL<<20, budget=1024ULL<<20;
    uvm_prefetch_replacement_record_population(&s,batch,batch/2,budget/4,budget,true);
    assert(s.request_stride==4 && s.short_quarter_bytes==batch/2);
    uvm_prefetch_replacement_record_population(&s,batch,batch/4,budget/4,budget,true);
    assert(s.request_stride==4); // exactly 25% holds
    uvm_prefetch_replacement_record_population(&s,batch,batch/4-1,budget/4,budget,true);
    assert(s.request_stride==1); // below 25% recovers
    uvm_prefetch_replacement_record_population(&s,batch,batch/2-1,budget/4,budget,true);
    assert(s.request_stride==1); // below 50% holds
    uvm_prefetch_replacement_record_population(&s,batch,batch,budget/4+1,budget,true);
    assert(s.request_stride==1 && s.last_short_quarter_bytes==0);
    assert(s.short_half_bytes==s.repopulated_bytes && s.short_budget_bytes==s.repopulated_bytes);
    assert(uvm_prefetch_replacement_batch(16ULL<<30)==1024ULL<<20);
}
int main(void) {
    low_churn_is_permissive(); high_churn_throttles_and_recovers();
    neighbors_do_not_feed_controller(); boundaries_zero_budget_and_saturation();
    epoch_quota_is_bounded_and_refreshes(); epoch_budget_is_frozen_then_resized();
    ratio_hysteresis_and_distance_boundaries(); return 0;
}
'''

def main():
    source = PRELUDE + f'#include "{HEADER}"\n' + CASES
    with tempfile.TemporaryDirectory() as d:
        c = Path(d) / "test.c"; exe = Path(d) / "test"
        c.write_text(source)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", "-g", str(c), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    print("prefetch replacement tests passed")

if __name__ == "__main__":
    main()
