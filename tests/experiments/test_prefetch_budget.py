#!/usr/bin/env python3
"""Execute the actual budget controller under deterministic concurrent schedules."""
from pathlib import Path
import re
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
static void sequential_run(uvm_prefetch_budget_t *b, NvU64 address, NvU64 now)
{
    for (unsigned i = 0; i < 4; ++i)
        uvm_prefetch_budget_observe(b, &config, 7, address + i * Q, now + i);
    assert(b->sequential_blocks == 4);
}

static void sequential_detection(void)
{
    uvm_prefetch_budget_t b = {0};
    sequential_run(&b, 0, 1);
    assert(!b.initialized && !b.success_pending && !b.window_bytes);
    uvm_prefetch_budget_observe(&b, &config, 7, 3 * Q, 5);
    assert(b.sequential_blocks == 4 && b.sequential_ns == 4); // Duplicate ignored.
    uvm_prefetch_budget_observe(&b, &config, 7, 2 * Q, 6);
    assert(b.sequential_blocks == 1); // Direction reversed.
    uvm_prefetch_budget_observe(&b, &config, 7, Q, 7);
    uvm_prefetch_budget_observe(&b, &config, 7, 0, 8);
    assert(b.sequential_blocks == 3 && b.sequential_reverse);
    uvm_prefetch_budget_observe(&b, &config, 8, Q, 9);
    assert(b.sequential_blocks == 1); // Different range.
    uvm_prefetch_budget_observe(&b, &config, 8, 4 * Q, 10);
    assert(b.sequential_blocks == 1); // Non-adjacent addresses.
    uvm_prefetch_budget_observe(&b, &config, 8, 5 * Q, 111);
    assert(b.sequential_blocks == 1); // Expired evidence.
    uvm_prefetch_budget_observe(&b, &config, 8, 6 * Q, 110);
    assert(b.sequential_blocks == 1); // Backwards clock.
    sequential_run(&b, 0, 120);
    uvm_prefetch_budget_backoff(&b, &config, 124);
    assert(!b.sequential_blocks);
    for (unsigned i = 0; i < 4; ++i)
        uvm_prefetch_budget_observe(&b, &config, 7, i * Q, 200 + i * 40);
    assert(b.sequential_blocks == 1); // Whole run, not only each gap, must fit.
}

static void expired_locality_uses_slow_growth(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t ticket;
    uvm_prefetch_budget_backoff(&b, &config, 0);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 1, &ticket));
    uvm_prefetch_budget_complete(&b, &config, &ticket, 2 * Q, Q, 2, true);
    sequential_run(&b, 0, 3);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 107, &ticket));
    assert(b.window_bytes == 9 * Q && !b.sequential_recoveries && b.recoveries == 1);
    uvm_prefetch_budget_complete(&b, &config, &ticket, 0, 0, 108, false);
}

static void sequential_growth_is_bounded_and_requires_copies(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t ticket;
    uvm_prefetch_budget_backoff(&b, &config, 0);  // 16MiB.
    uvm_prefetch_budget_backoff(&b, &config, 10); // 8MiB, quiet deadline 110.
    sequential_run(&b, 0, 11);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 15, &ticket));
    assert(b.window_bytes == 4 * Q && !b.recoveries); // Hint alone cannot grow.
    uvm_prefetch_budget_complete(&b, &config, &ticket, Q, 0, 16, true);
    assert(!b.success_pending); // Trigger-only movement cannot grow either.
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 17, &ticket));
    uvm_prefetch_budget_complete(&b, &config, &ticket, 2 * Q, Q, 18, true);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 109, &ticket));
    assert(b.window_bytes == 4 * Q); // Locality cannot bypass negative deadline.
    uvm_prefetch_budget_complete(&b, &config, &ticket, 0, 0, 109, false);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 110, &ticket));
    assert(b.window_bytes == 8 * Q && b.sequential_recoveries == 1);
    assert(!b.sequential_blocks && !b.success_pending); // Consume both credits.
    uvm_prefetch_budget_complete(&b, &config, &ticket, 2 * Q, Q, 111, true);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 121, &ticket));
    assert(b.window_bytes == 8 * Q); // New copies require a fresh run for fast growth.
    uvm_prefetch_budget_complete(&b, &config, &ticket, 0, 0, 122, false);
    sequential_run(&b, 8 * Q, 123);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 127, &ticket));
    assert(b.window_bytes == 16 * Q && b.sequential_recoveries == 2);
    assert(b.window_bytes == config.limit_bytes && b.reserved_bytes <= b.window_bytes);
    uvm_prefetch_budget_complete(&b, &config, &ticket, 2 * Q, Q, 128, true);
    assert(!b.success_pending && !b.reserved_bytes);
}

static void sequential_default_timing_and_range_isolation(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t ticket;
    uvm_prefetch_budget_config_t cfg = {DEFAULT_LIMIT_MB * Q / 2, 0,
                                         DEFAULT_BACKOFF_MS, DEFAULT_RECOVERY_MS};
    uvm_prefetch_budget_backoff(&b, &cfg, 0);
    uvm_prefetch_budget_backoff(&b, &cfg, 50);
    assert(b.window_bytes == 4 * Q && b.next_growth_ns == 550);
    assert(uvm_prefetch_budget_reserve(&b, &cfg, 7, 2 * Q, 64 * Q, 60, &ticket));
    uvm_prefetch_budget_complete(&b, &cfg, &ticket, 2 * Q, Q, 61, true);
    for (unsigned i = 0; i < 4; ++i)
        uvm_prefetch_budget_observe(&b, &cfg, 8, i * Q, 550 + i);
    assert(uvm_prefetch_budget_reserve(&b, &cfg, 7, 2 * Q, 64 * Q, 553, &ticket));
    assert(b.window_bytes == 4 * Q && !b.recoveries); // Other allocation's hint cannot accelerate.
    uvm_prefetch_budget_complete(&b, &cfg, &ticket, 0, 0, 553, false);
    for (unsigned i = 0; i < 4; ++i)
        uvm_prefetch_budget_observe(&b, &cfg, 7, i * Q, 554 + i);
    assert(uvm_prefetch_budget_reserve(&b, &cfg, 7, 2 * Q, 64 * Q, 557, &ticket));
    assert(b.window_bytes == 8 * Q && b.sequential_recoveries == 1);
    uvm_prefetch_budget_complete(&b, &cfg, &ticket, 2 * Q, Q, 558, true);
    for (unsigned i = 0; i < 4; ++i)
        uvm_prefetch_budget_observe(&b, &cfg, 7, (i + 4) * Q, 560 + i);
    assert(uvm_prefetch_budget_reserve(&b, &cfg, 7, 2 * Q, 64 * Q, 607, &ticket));
    assert(b.window_bytes == 8 * Q); // 49ms since successful copy: no growth.
    uvm_prefetch_budget_complete(&b, &cfg, &ticket, 0, 0, 607, false);
    assert(uvm_prefetch_budget_reserve(&b, &cfg, 7, 2 * Q, 64 * Q, 608, &ticket));
    assert(b.window_bytes == 16 * Q && b.sequential_recoveries == 2); // Exact 50ms.
    uvm_prefetch_budget_complete(&b, &cfg, &ticket, 0, 0, 609, false);
    assert(!b.reserved_bytes);
}

static void negative_cancels_sequential_credit(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t ticket;
    uvm_prefetch_budget_backoff(&b, &config, 0);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 1, &ticket));
    uvm_prefetch_budget_complete(&b, &config, &ticket, 2 * Q, Q, 2, true);
    sequential_run(&b, 0, 3);
    assert(b.success_pending && b.sequential_blocks == 4);
    uvm_prefetch_budget_backoff(&b, &config, 7); // Reduction rate limited.
    assert(b.window_bytes == 8 * Q && b.backoffs == 1);
    assert(!b.success_pending && !b.sequential_blocks && b.next_growth_ns == 107);
    sequential_run(&b, 4 * Q, 8);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 12, &ticket));
    uvm_prefetch_budget_complete(&b, &config, &ticket, 2 * Q, Q, 13, true);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 106, &ticket));
    assert(b.window_bytes == 8 * Q && !b.sequential_recoveries);
    uvm_prefetch_budget_complete(&b, &config, &ticket, 0, 0, 106, false);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 107, &ticket));
    assert(b.window_bytes == 16 * Q && b.sequential_recoveries == 1);
    uvm_prefetch_budget_complete(&b, &config, &ticket, 0, 0, 108, false);
}

static void default_backoff_tolerates_short_bursts(void)
{
    // Use the actual module defaults, with clock units in milliseconds.
    uvm_prefetch_budget_config_t cfg = {DEFAULT_LIMIT_MB * Q / 2, 0,
                                         DEFAULT_BACKOFF_MS, DEFAULT_RECOVERY_MS};
    uvm_prefetch_budget_t b = {0};
    assert(cfg.backoff_ns == 50);
    uvm_prefetch_budget_backoff(&b, &cfg, 0);
    assert(b.window_bytes == 8 * Q);
    for (unsigned t = 1; t < 150; ++t) {
        uvm_prefetch_budget_backoff(&b, &cfg, t);
        if (t < 50)
            assert(b.window_bytes == 8 * Q);
        else if (t < 100)
            assert(b.window_bytes == 4 * Q);
        else
            assert(b.window_bytes == 2 * Q);
    }
    assert(b.backoffs == 3 && b.feedback_generation == 150);
    uvm_prefetch_budget_backoff(&b, &cfg, 150);
    assert(!b.window_bytes && b.backoffs == 4);
    assert(b.next_growth_ns == 150 + DEFAULT_RECOVERY_MS);
}

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
    assert(uvm_prefetch_budget_reserve(&b, &cfg, 7, 3 * Q + Q / 2, 16 * Q, 0, &a) == 3 * Q);
    assert(uvm_prefetch_budget_reserve(&b, &cfg, 7, 3 * Q, 16 * Q, 1, &c) == 2 * Q);
    assert(b.reserved_bytes == 5 * Q && b.peak_bytes == 5 * Q);
    assert(!uvm_prefetch_budget_reserve(&b, &cfg, 7, Q, 16 * Q, 2, &denied));
    assert(!denied.bytes && !b.backoffs && !b.capacity_stops);
    uvm_prefetch_budget_complete(&b, &cfg, &a, a.bytes, Q, 3, true);
    uvm_prefetch_budget_complete(&b, &cfg, &c, 0, 0, 4, false);
    assert(!b.reserved_bytes && !b.success_pending); // Already at hard cap.

    memset(&b, 0, sizeof(b));
    cfg = config;
    cfg.demand_reserve_bytes = 2 * Q;
    assert(!uvm_prefetch_budget_reserve(&b, &cfg, 7, Q, 2 * Q, 0, &a));
    assert(b.capacity_stops == 1 && b.backoffs == 1 && !b.reserved_bytes);
    memset(&b, 0, sizeof(b));
    assert(!uvm_prefetch_budget_reserve(&b, &config, 7, Q, 0, 0, &a));
    assert(b.capacity_stops == 1);
    for (unsigned i = 0; i < 2; ++i) {
        memset(&b, 0, sizeof(b));
        cfg = config;
        cfg.limit_bytes = i ? Q / 2 : 0;
        assert(!uvm_prefetch_budget_reserve(&b, &cfg, 7, 4 * Q, 64 * Q, 0, &a));
        assert(!b.reserved_bytes && !b.initialized);
    }
}

static void drain_then_single_probe_and_reopen(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t old, probe, denied, reopened;
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 4 * Q, 64 * Q, 0, &old) == 4 * Q);
    for (unsigned t = 1; t <= 31; t += 10)
        uvm_prefetch_budget_backoff(&b, &config, t);
    assert(!b.window_bytes && b.reserved_bytes == 4 * Q);
    assert(!uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 131, &denied));
    uvm_prefetch_budget_complete(&b, &config, &old, 4 * Q, Q, 132, true);
    assert(!b.success_pending); // Old generation cannot revive OFF.
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 8 * Q, 64 * Q, 133, &probe) == 2 * Q);
    assert(probe.probe && b.probe_inflight && !b.window_bytes);
    NvU64 generation = b.feedback_generation;
    assert(!uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 134, &denied));
    assert(b.probes == 1 && b.feedback_generation == generation);
    uvm_prefetch_budget_complete(&b, &config, &probe, 2 * Q, Q, 140, true);
    assert(!b.probe_inflight && !b.window_bytes && b.success_pending);
    sequential_run(&b, 0, 141);
    assert(!uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 150, &denied));
    assert(!b.window_bytes); // Even a sequential run cannot skip OFF observation.
    assert(!uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 239, &denied));
    assert(!b.recoveries);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 8 * Q, 64 * Q, 240, &reopened) == 2 * Q);
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
        assert(!uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 129, &denied));
        assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 130, &probe) == 2 * Q);
        if (outcome == 3)
            uvm_prefetch_budget_backoff(&b, &config, 135);
        // failure, empty success, trigger-only success, and invalidated success.
        NvU64 moved = outcome == 1 ? 0 : 2 * Q;
        NvU64 neighbors = outcome >= 3 ? Q : 0;
        uvm_prefetch_budget_complete(&b, &config, &probe, moved, neighbors, 140, outcome != 0);
        assert(!b.window_bytes && !b.success_pending && b.probe_failures == 1);
        assert(!uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 239, &denied));
        assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 240, &probe) == 2 * Q);
        assert(probe.probe && b.probes == 2);
        uvm_prefetch_budget_complete(&b, &config, &probe, 0, 0, 241, false);
    }
}

static void feedback_cancels_passed_probe(void)
{
    uvm_prefetch_budget_t b;
    uvm_prefetch_reservation_t probe, denied;
    off(&b);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 130, &probe) == 2 * Q);
    uvm_prefetch_budget_complete(&b, &config, &probe, 2 * Q, Q, 140, true);
    uvm_prefetch_budget_backoff(&b, &config, 150);
    assert(!b.success_pending && !b.window_bytes);
    assert(!uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 249, &denied));
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 250, &probe) == 2 * Q);
    assert(probe.probe && !b.recoveries);
    uvm_prefetch_budget_complete(&b, &config, &probe, 0, 0, 251, false);
}

static void regular_growth_needs_observation(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t a, c, d;
    uvm_prefetch_budget_backoff(&b, &config, 0);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 1, &a) == 2 * Q);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 2, &c) == 2 * Q);
    uvm_prefetch_budget_complete(&b, &config, &a, 2 * Q, Q, 10, true);
    uvm_prefetch_budget_complete(&b, &config, &c, 2 * Q, Q, 50, true);
    assert(b.success_pending && b.success_ns == 10 && !b.recoveries);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 109, &d) == 2 * Q);
    assert(b.window_bytes == 8 * Q && !b.recoveries);
    uvm_prefetch_budget_complete(&b, &config, &d, 0, 0, 109, false);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 110, &d) == 2 * Q);
    assert(b.window_bytes == 9 * Q && b.recoveries == 1 && !b.success_pending);
    uvm_prefetch_budget_complete(&b, &config, &d, 0, 0, 111, false);
}

static void rate_limited_feedback_invalidates_old_work(void)
{
    uvm_prefetch_budget_t b = {0};
    uvm_prefetch_reservation_t a, c;
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 0, &a) == 2 * Q);
    uvm_prefetch_budget_backoff(&b, &config, 5);
    assert(b.window_bytes == 8 * Q);
    uvm_prefetch_budget_complete(&b, &config, &a, 2 * Q, Q, 6, true);
    assert(!b.success_pending);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 7, &c) == 2 * Q);
    uvm_prefetch_budget_complete(&b, &config, &c, 2 * Q, Q, 8, true);
    assert(b.success_pending);
    uvm_prefetch_budget_backoff(&b, &config, 10); // Too soon to halve again.
    assert(b.window_bytes == 8 * Q && b.backoffs == 1 && !b.success_pending);
    assert(b.feedback_generation == 2 && b.next_growth_ns == 110);
    assert(uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, 64 * Q, 111, &c) == 2 * Q);
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
    assert(!uvm_prefetch_budget_reserve(&b, &cfg, 7, Q, 64 * Q, 1000, &ticket));
    assert(!b.probes && !b.window_bytes);
    off(&b);
    assert(!uvm_prefetch_budget_reserve(&b, &config, 7, Q, 64 * Q, 130, &ticket));
    assert(!b.probes);
    assert(!uvm_prefetch_budget_reserve(&b, &config, 7, 2 * Q, Q, 131, &ticket));
    assert(!b.probes && b.capacity_stops == 1 && b.next_growth_ns == 231);
}

int main(void)
{
    default_backoff_tolerates_short_bursts();
    sequential_detection();
    expired_locality_uses_slow_growth();
    sequential_growth_is_bounded_and_requires_copies();
    sequential_default_timing_and_range_isolation();
    negative_cancels_sequential_credit();
    caps_and_capacity();
    drain_then_single_probe_and_reopen();
    rejected_probes_have_cooldown();
    feedback_cancels_passed_probe();
    regular_growth_needs_observation();
    rate_limited_feedback_invalidates_old_work();
    no_illegal_or_partial_probe();
    puts("Production budget: OFF, probes, deferred/sequential recovery, epochs, cooldowns and aggregate caps passed.");
    return 0;
}
'''


def main():
    header = (ROOT / 'kernel-open/nvidia-uvm/uvm_prefetch_budget.h').read_text()
    pmm = (ROOT / 'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text()
    defaults = ''
    for macro, parameter in [('LIMIT_MB', 'limit_mb'), ('BACKOFF_MS', 'backoff_ms'),
                             ('RECOVERY_MS', 'recovery_ms')]:
        value = re.search(r'static unsigned uvm_perf_speculative_prefetch_' + parameter
                          + r' = (\d+);', pmm).group(1)
        defaults += f'#define DEFAULT_{macro} {value}U\n'
    code = PRELUDE + defaults + header.replace('#include "nvtypes.h"', '') + CASES
    with tempfile.TemporaryDirectory(prefix='prefetch-budget-') as directory:
        c = Path(directory) / 'test.c'
        binary = Path(directory) / 'test'
        c.write_text(code)
        subprocess.run(['cc', '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=address,undefined', '-g', str(c), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
