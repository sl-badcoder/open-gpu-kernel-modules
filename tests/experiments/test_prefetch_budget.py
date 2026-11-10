#!/usr/bin/env python3
"""Execute the actual budget controller under deterministic concurrent schedules."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).parents[2]
PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef uint64_t NvU64;
#define UVM_ASSERT assert
'''

CASES = r'''
#define Q UVM_PREFETCH_BUDGET_QUANTUM
// Abstract clock ticks: reduction spacing 10, recovery/cooldown 100.
static uvm_prefetch_budget_config_t config = {16 * Q, 0, 10, 100};
static void off(uvm_prefetch_budget_t *b)
{
    memset(b, 0, sizeof(*b));
    uvm_prefetch_budget_backoff(b, &config, 0);
    assert(b->window_bytes == 8 * Q);
    uvm_prefetch_budget_backoff(b, &config, 10);
    assert(b->window_bytes == 4 * Q);
    uvm_prefetch_budget_backoff(b, &config, 20);
    assert(b->window_bytes == 2 * Q);
    uvm_prefetch_budget_backoff(b, &config, 30);
    assert(b->initialized && !b->window_bytes && b->next_growth_ns == 130);
}

static void caps_and_capacity(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t a, c, denied;
    uvm_prefetch_budget_config_t cfg = {5 * Q + 1, Q, 10, 100};
    assert(uvm_prefetch_budget_reserve(&b, &cfg, 3 * Q + Q / 2, 16 * Q, 0, &a) == 3 * Q);
    assert(uvm_prefetch_budget_reserve(&b, &cfg, 3 * Q, 16 * Q, 1, &c) == 2 * Q);
    assert(b.reserved_bytes == 5 * Q && b.peak_bytes == 5 * Q);
    assert(!uvm_prefetch_budget_reserve(&b, &cfg, Q, 16 * Q, 2, &denied));
    assert(!denied.bytes && !b.backoffs && !b.capacity_stops);
    uvm_prefetch_budget_complete(&b, &cfg, &a, a.bytes, Q, 3, true);
    uvm_prefetch_budget_complete(&b, &cfg, &c, 0, 0, 4, false);
    assert(!b.reserved_bytes && !b.success_pending); // Already at hard cap.

    memset(&b, 0, sizeof(b));
    cfg = config;
    cfg.demand_reserve_bytes = 2 * Q;
    assert(!uvm_prefetch_budget_reserve(&b, &cfg, Q, 2 * Q, 0, &a));
    assert(b.capacity_stops == 1 && b.backoffs == 1 && !b.reserved_bytes);
    memset(&b, 0, sizeof(b));
    assert(!uvm_prefetch_budget_reserve(&b, &config, Q, 0, 0, &a));
    assert(b.capacity_stops == 1);
    for (unsigned i = 0; i < 2; ++i) {
        memset(&b, 0, sizeof(b));
        cfg = config;
        cfg.limit_bytes = i ? Q / 2 : 0;
        assert(!uvm_prefetch_budget_reserve(&b, &cfg, 4 * Q, 64 * Q, 0, &a));
        assert(!b.reserved_bytes && !b.initialized);
    }
}

static void drain_then_single_probe_and_reopen(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t old, probe, denied, reopened;
    assert(uvm_prefetch_budget_reserve(&b, &config, 4 * Q, 64 * Q, 0, &old) == 4 * Q);
    for (unsigned t = 1; t <= 31; t += 10)
        uvm_prefetch_budget_backoff(&b, &config, t);
    assert(!b.window_bytes && b.reserved_bytes == 4 * Q);
    assert(!uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 131, &denied));
    uvm_prefetch_budget_complete(&b, &config, &old, 4 * Q, Q, 132, true);
    assert(!b.success_pending); // Old generation cannot revive OFF.
    assert(uvm_prefetch_budget_reserve(&b, &config, 8 * Q, 64 * Q, 133, &probe) == 2 * Q);
    assert(probe.probe && b.probe_inflight && !b.window_bytes);
    NvU64 generation = b.feedback_generation;
    assert(!uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 134, &denied));
    assert(b.probes == 1 && b.feedback_generation == generation);
    uvm_prefetch_budget_complete(&b, &config, &probe, 2 * Q, Q, 140, true);
    assert(!b.probe_inflight && !b.window_bytes && b.success_pending);
    assert(!uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 239, &denied));
    assert(!b.recoveries);
    assert(uvm_prefetch_budget_reserve(&b, &config, 8 * Q, 64 * Q, 240, &reopened) == 2 * Q);
    assert(!reopened.probe && b.window_bytes == 2 * Q && b.recoveries == 1);
    uvm_prefetch_budget_complete(&b, &config, &reopened, 0, 0, 241, false);
    assert(!b.reserved_bytes);
}

static void rejected_probes_have_cooldown(void)
{
    for (unsigned outcome = 0; outcome < 4; ++outcome) {
        uvm_prefetch_budget_t b;
        uvm_prefetch_reservation_t probe, denied;
        off(&b);
        assert(!uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 129, &denied));
        assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 130, &probe) == 2 * Q);
        if (outcome == 3)
            uvm_prefetch_budget_backoff(&b, &config, 135);
        // failure, empty success, trigger-only success, and invalidated success.
        NvU64 moved = outcome == 1 ? 0 : 2 * Q;
        NvU64 neighbors = outcome >= 3 ? Q : 0;
        uvm_prefetch_budget_complete(&b, &config, &probe, moved, neighbors, 140, outcome != 0);
        assert(!b.window_bytes && !b.success_pending && b.probe_failures == 1);
        assert(!uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 239, &denied));
        assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 240, &probe) == 2 * Q);
        assert(probe.probe && b.probes == 2);
        uvm_prefetch_budget_complete(&b, &config, &probe, 0, 0, 241, false);
    }
}

static void feedback_cancels_passed_probe(void)
{
    uvm_prefetch_budget_t b;
    uvm_prefetch_reservation_t probe, denied;
    off(&b);
    assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 130, &probe) == 2 * Q);
    uvm_prefetch_budget_complete(&b, &config, &probe, 2 * Q, Q, 140, true);
    uvm_prefetch_budget_backoff(&b, &config, 150);
    assert(!b.success_pending && !b.window_bytes);
    assert(!uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 249, &denied));
    assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 250, &probe) == 2 * Q);
    assert(probe.probe && !b.recoveries);
    uvm_prefetch_budget_complete(&b, &config, &probe, 0, 0, 251, false);
}

static void regular_growth_needs_observation(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t a, c, d;
    uvm_prefetch_budget_backoff(&b, &config, 0);
    assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 1, &a) == 2 * Q);
    assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 2, &c) == 2 * Q);
    uvm_prefetch_budget_complete(&b, &config, &a, 2 * Q, Q, 10, true);
    uvm_prefetch_budget_complete(&b, &config, &c, 2 * Q, Q, 50, true);
    assert(b.success_pending && b.success_ns == 10 && !b.recoveries);
    assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 109, &d) == 2 * Q);
    assert(b.window_bytes == 8 * Q && !b.recoveries);
    uvm_prefetch_budget_complete(&b, &config, &d, 0, 0, 109, false);
    assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 110, &d) == 2 * Q);
    assert(b.window_bytes == 9 * Q && b.recoveries == 1 && !b.success_pending);
    uvm_prefetch_budget_complete(&b, &config, &d, 0, 0, 111, false);
}

static void rate_limited_feedback_invalidates_old_work(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t a, c;
    assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 0, &a) == 2 * Q);
    uvm_prefetch_budget_backoff(&b, &config, 5);
    assert(b.window_bytes == 8 * Q);
    uvm_prefetch_budget_complete(&b, &config, &a, 2 * Q, Q, 6, true);
    assert(!b.success_pending);
    assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 7, &c) == 2 * Q);
    uvm_prefetch_budget_complete(&b, &config, &c, 2 * Q, Q, 8, true);
    assert(b.success_pending);
    uvm_prefetch_budget_backoff(&b, &config, 10); // Too soon to halve again.
    assert(b.window_bytes == 8 * Q && b.backoffs == 1 && !b.success_pending);
    assert(b.feedback_generation == 2 && b.next_growth_ns == 110);
    assert(uvm_prefetch_budget_reserve(&b, &config, 2 * Q, 64 * Q, 111, &c) == 2 * Q);
    assert(!b.recoveries);
    uvm_prefetch_budget_complete(&b, &config, &c, 0, 0, 112, false);
}

static void no_illegal_or_partial_probe(void)
{
    uvm_prefetch_budget_t b;
    uvm_prefetch_reservation_t ticket;
    uvm_prefetch_budget_config_t cfg = config;
    memset(&b, 0, sizeof(b));
    cfg.limit_bytes = Q;
    uvm_prefetch_budget_backoff(&b, &cfg, 0);
    assert(!uvm_prefetch_budget_reserve(&b, &cfg, Q, 64 * Q, 1000, &ticket));
    assert(!b.probes && !b.window_bytes);
    off(&b);
    assert(!uvm_prefetch_budget_reserve(&b, &config, Q, 64 * Q, 130, &ticket));
    assert(!b.probes);
    assert(!uvm_prefetch_budget_reserve(&b, &config, 2 * Q, Q, 131, &ticket));
    assert(!b.probes && b.capacity_stops == 1 && b.next_growth_ns == 231);
}

int main(void)
{
    caps_and_capacity();
    drain_then_single_probe_and_reopen();
    rejected_probes_have_cooldown();
    feedback_cancels_passed_probe();
    regular_growth_needs_observation();
    rate_limited_feedback_invalidates_old_work();
    no_illegal_or_partial_probe();
    puts("Production budget: OFF, single probes, deferred recovery, epochs, cooldowns and aggregate caps passed.");
    return 0;
}
'''


def main():
    header = (ROOT / 'kernel-open/nvidia-uvm/uvm_prefetch_budget.h').read_text()
    code = PRELUDE + header.replace('#include "nvtypes.h"', '') + CASES
    with tempfile.TemporaryDirectory(prefix='prefetch-budget-') as directory:
        c = Path(directory) / 'test.c'
        binary = Path(directory) / 'test'
        c.write_text(code)
        subprocess.run(['cc', '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=address,undefined', '-g', str(c), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
