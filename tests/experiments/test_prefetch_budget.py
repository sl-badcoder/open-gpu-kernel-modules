#!/usr/bin/env python3
"""Behavioral tests for the production speculative-prefetch byte budget."""
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
#include <limits.h>
typedef uint64_t NvU64;
#define UVM_ASSERT assert
'''

SCENARIOS = r'''
static const NvU64 Q = UVM_PREFETCH_BUDGET_QUANTUM;
static void fresh(uvm_prefetch_budget_t *b) { memset(b, 0, sizeof(*b)); }
int main(void)
{
    uvm_prefetch_budget_t b, c;
    NvU64 grant;

    /* Disabled/sub-block budgets, absent free capacity, and concurrent cap exhaustion. */
    fresh(&b);
    assert(!uvm_prefetch_budget_reserve(&b, Q, 8 * Q, 0, 0, 10, 100));
    assert(!uvm_prefetch_budget_reserve(&b, Q, 8 * Q, Q / 2, 0, 10, 100));
    assert(!uvm_prefetch_budget_reserve(&b, Q - 1, 8 * Q, Q, 0, 10, 100));
    assert(!uvm_prefetch_budget_reserve(&b, Q, 0, Q, 0, 10, 100));
    fresh(&b);
    for (unsigned i = 0; i < 6; ++i) {
        grant = uvm_prefetch_budget_reserve(&b, Q, 16 * Q, 4 * Q, 0, 10, 100);
        assert(grant == (i < 4 ? Q : 0));
        assert(b.reserved_bytes <= 4 * Q);
    }
    uvm_prefetch_budget_complete(&b, 4 * Q, 4 * Q, 4 * Q, 10, 100, false);
    assert(!b.reserved_bytes);

    /* Limit and request sizes are rounded down to full VA-block quanta. */
    assert(uvm_prefetch_budget_limit(Q + 1) == Q);
    fresh(&b);
    grant = uvm_prefetch_budget_reserve(&b, 3 * Q + Q / 2, 8 * Q, 5 * Q,
                                        Q, 100, 10);
    assert(grant == 3 * Q && b.reserved_bytes == grant);
    assert(uvm_prefetch_budget_reserve(&b, Q, 8 * Q, 5 * Q, Q, 101, 10) == Q);
    assert(b.reserved_bytes == 4 * Q && b.peak_bytes == 4 * Q);

    /* Demand reserve, no free capacity, and a full cap deny admission. */
    assert(uvm_prefetch_budget_reserve(&b, Q, 2 * Q, 5 * Q, 2 * Q, 102, 10) == 0);
    assert(b.capacity_stops == 1 && b.denials == 1);
    /* Backoff applies to aggregate outstanding reservations; drain before retrying. */
    assert(uvm_prefetch_budget_reserve(&b, Q, 8 * Q, 5 * Q, Q, 103, 10) == 0);
    uvm_prefetch_budget_complete(&b, 4 * Q, 0, 5 * Q, 104, 10, false);
    assert(uvm_prefetch_budget_reserve(&b, Q, 8 * Q, 5 * Q, Q, 105, 10) == Q);

    /* Interleaved reservations cannot exceed the cap, and completion is safe after errors. */
    fresh(&b); fresh(&c);
    assert(uvm_prefetch_budget_reserve(&b, 2 * Q, 8 * Q, 4 * Q, 0, 10, 100) == 2 * Q);
    assert(uvm_prefetch_budget_reserve(&c, 3 * Q, 8 * Q, 4 * Q, 0, 11, 100) == 3 * Q);
    assert(b.reserved_bytes == 2 * Q && c.reserved_bytes == 3 * Q);
    uvm_prefetch_budget_complete(&b, 2 * Q, 0, 4 * Q, 12, 100, false);
    assert(b.reserved_bytes == 0 && b.recoveries == 0);
    uvm_prefetch_budget_complete(&c, 3 * Q, Q, 4 * Q, 13, 100, true);
    assert(c.reserved_bytes == 0 && c.recoveries == 0);

    /* Backoff halves the window, is rate limited, and retains a neighbor probe. */
    fresh(&b);
    assert(uvm_prefetch_budget_reserve(&b, 4 * Q, 8 * Q, 8 * Q, 0, 20, 100) == 4 * Q);
    uvm_prefetch_budget_backoff(&b, 8 * Q, 30, 100);
    assert(b.window_bytes == 4 * Q && b.backoffs == 1);
    uvm_prefetch_budget_backoff(&b, 8 * Q, 50, 100);
    assert(b.window_bytes == 4 * Q && b.backoffs == 1);
    uvm_prefetch_budget_backoff(&b, 8 * Q, 131, 100);
    assert(b.window_bytes == 2 * Q && b.backoffs == 2);
    uvm_prefetch_budget_backoff(&b, 8 * Q, 232, 100);
    assert(b.window_bytes == 2 * Q && b.backoffs == 3);
    fresh(&c);
    uvm_prefetch_budget_backoff(&c, Q, 100, 50);
    assert(c.window_bytes == Q); /* A smaller hard cap is always respected. */

    /* Only successful non-empty transfers after quiet time recover one quantum. */
    fresh(&b);
    uvm_prefetch_budget_backoff(&b, 4 * Q, 100, 50);
    assert(b.window_bytes == 2 * Q);
    uvm_prefetch_budget_complete(&b, 0, 0, 4 * Q, 151, 50, true);
    assert(b.window_bytes == 2 * Q && b.recoveries == 0);
    grant = uvm_prefetch_budget_reserve(&b, Q, 4 * Q, 4 * Q, 0, 152, 50);
    assert(grant == Q);
    uvm_prefetch_budget_complete(&b, Q, Q, 4 * Q, 152, 50, false);
    assert(b.window_bytes == 2 * Q && b.recoveries == 0);
    grant = uvm_prefetch_budget_reserve(&b, Q, 4 * Q, 4 * Q, 0, 153, 50);
    assert(grant == Q);
    uvm_prefetch_budget_complete(&b, Q, Q, 4 * Q, 153, 50, true);
    assert(b.window_bytes == 3 * Q && b.recoveries == 1);

    /* Rate-limited bad feedback still postpones recovery until truly quiet. */
    fresh(&b);
    uvm_prefetch_budget_backoff(&b, 4 * Q, 100, 50);
    uvm_prefetch_budget_backoff(&b, 4 * Q, 140, 50);
    assert(b.backoffs == 1 && b.next_growth_ns == 190);
    grant = uvm_prefetch_budget_reserve(&b, Q, 8 * Q, 4 * Q, 0, 151, 50);
    uvm_prefetch_budget_complete(&b, grant, grant, 4 * Q, 151, 50, true);
    assert(b.window_bytes == 2 * Q && !b.recoveries);
    grant = uvm_prefetch_budget_reserve(&b, Q, 8 * Q, 4 * Q, 0, 190, 50);
    uvm_prefetch_budget_complete(&b, grant, grant, 4 * Q, 190, 50, true);
    assert(b.window_bytes == 3 * Q && b.recoveries == 1);
    grant = uvm_prefetch_budget_reserve(&b, Q, 4 * Q, 4 * Q, 0, 191, 50);
    assert(grant == Q);
    uvm_prefetch_budget_complete(&b, Q, Q, 4 * Q, 191, 50, true);
    assert(b.window_bytes == 3 * Q && b.recoveries == 1);

    puts("Prefetch budget cap, rounding, reservations, backoff, quiet rate limit and recovery passed.");
    return 0;
}
'''

def main():
    header = (ROOT / "kernel-open/nvidia-uvm/uvm_prefetch_budget.h").read_text()
    code = PRELUDE + header.replace('#include "nvtypes.h"', '') + SCENARIOS
    with tempfile.TemporaryDirectory(prefix="prefetch-budget-") as directory:
        c = Path(directory) / "test.c"
        binary = Path(directory) / "test"
        c.write_text(code)
        subprocess.run(["cc", "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", "-g", str(c), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

if __name__ == "__main__":
    main()
