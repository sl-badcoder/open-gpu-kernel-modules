/* SPDX-License-Identifier: MIT */
#ifndef UVM_PREFETCH_BUDGET_H
#define UVM_PREFETCH_BUDGET_H

#include "nvtypes.h"

// Reserve full VA blocks, including sparse blocks, so submitted migration
// bytes cannot exceed the admission. All operations require PMM list_lock.
#define UVM_PREFETCH_BUDGET_QUANTUM ((NvU64)2 * 1024 * 1024)

typedef struct {
    NvU64 reserved_bytes;
    NvU64 peak_bytes;
    NvU64 window_bytes;
    NvU64 last_backoff_ns;
    NvU64 next_growth_ns;
    NvU64 admissions;
    NvU64 denials;
    NvU64 backoffs;
    NvU64 quick_returns;
    NvU64 capacity_stops;
    NvU64 recoveries;
} uvm_prefetch_budget_t;

static inline NvU64 uvm_prefetch_budget_limit(NvU64 bytes)
{
    return bytes - bytes % UVM_PREFETCH_BUDGET_QUANTUM;
}

static inline void uvm_prefetch_budget_backoff(uvm_prefetch_budget_t *budget,
                                              NvU64 limit, NvU64 now, NvU64 quiet_ns)
{
    NvU64 window, floor;

    limit = uvm_prefetch_budget_limit(limit);
    if (!limit)
        return;
    // Every negative observation postpones recovery, even when reduction is
    // rate limited. Sustained pressure must not look like a quiet interval.
    budget->next_growth_ns = now + quiet_ns;
    if (budget->last_backoff_ns && now >= budget->last_backoff_ns &&
        now - budget->last_backoff_ns < quiet_ns)
        return;
    window = budget->window_bytes ? budget->window_bytes : limit;
    if (window > limit)
        window = limit;
    window = uvm_prefetch_budget_limit(window / 2);
    // The trigger may already be resident after demand promotion. Keep room
    // for one neighbor probe, unless the configured hard cap forbids it.
    floor = limit < 2 * UVM_PREFETCH_BUDGET_QUANTUM ? limit : 2 * UVM_PREFETCH_BUDGET_QUANTUM;
    budget->window_bytes = window < floor ? floor : window;
    budget->last_backoff_ns = now;
    ++budget->backoffs;
}

static inline NvU64 uvm_prefetch_budget_reserve(uvm_prefetch_budget_t *budget,
                                               NvU64 requested, NvU64 free_bytes,
                                               NvU64 limit, NvU64 demand_reserve,
                                               NvU64 now, NvU64 quiet_ns)
{
    NvU64 grant, headroom;

    limit = uvm_prefetch_budget_limit(limit);
    if (!limit || requested < UVM_PREFETCH_BUDGET_QUANTUM)
        goto denied;
    if (!budget->window_bytes || budget->window_bytes > limit)
        budget->window_bytes = limit;
    if (budget->reserved_bytes >= budget->window_bytes)
        goto denied;

    // PMA may already exclude some reserved transfers. Subtracting the entire
    // reservation again is deliberately conservative, never optimistic.
    headroom = free_bytes > demand_reserve ? free_bytes - demand_reserve : 0;
    headroom = headroom > budget->reserved_bytes ? headroom - budget->reserved_bytes : 0;
    if (headroom < UVM_PREFETCH_BUDGET_QUANTUM) {
        ++budget->capacity_stops;
        uvm_prefetch_budget_backoff(budget, limit, now, quiet_ns);
        goto denied;
    }
    grant = requested;
    if (grant > budget->window_bytes)
        grant = budget->window_bytes;
    if (grant > budget->window_bytes - budget->reserved_bytes)
        grant = budget->window_bytes - budget->reserved_bytes;
    if (grant > headroom)
        grant = headroom;
    grant = uvm_prefetch_budget_limit(grant);
    if (!grant)
        goto denied;
    budget->reserved_bytes += grant;
    if (budget->reserved_bytes > budget->peak_bytes)
        budget->peak_bytes = budget->reserved_bytes;
    ++budget->admissions;
    return grant;

denied:
    ++budget->denials;
    return 0;
}

static inline void uvm_prefetch_budget_complete(uvm_prefetch_budget_t *budget,
                                                NvU64 reservation, NvU64 transferred,
                                                NvU64 limit, NvU64 now, NvU64 quiet_ns,
                                                bool success)
{
    UVM_ASSERT(reservation <= budget->reserved_bytes);
    UVM_ASSERT(transferred <= reservation);
    budget->reserved_bytes -= reservation;
    limit = uvm_prefetch_budget_limit(limit);
    // Recovery is a bounded probe after a quiet period, not evidence of reuse.
    // Failed or empty attempts never expand the speculative window.
    if (success && transferred && budget->window_bytes < limit && now >= budget->next_growth_ns) {
        NvU64 increase = limit - budget->window_bytes;
        if (increase > UVM_PREFETCH_BUDGET_QUANTUM)
            increase = UVM_PREFETCH_BUDGET_QUANTUM;
        budget->window_bytes += increase;
        budget->next_growth_ns = now + quiet_ns;
        ++budget->recoveries;
    }
}

#endif
