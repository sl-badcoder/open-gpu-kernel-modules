/* SPDX-License-Identifier: MIT */
#ifndef UVM_PREFETCH_BUDGET_H
#define UVM_PREFETCH_BUDGET_H

#include "nvtypes.h"

// Full-block reservations bound submitted payload even for sparse VA blocks.
// Every operation, including ticket creation, requires PMM list_lock.
#define UVM_PREFETCH_BUDGET_QUANTUM ((NvU64)2 * 1024 * 1024)

typedef struct {
    NvU64 limit_bytes;
    NvU64 demand_reserve_bytes;
    NvU64 backoff_ns;
    NvU64 recovery_ns;
} uvm_prefetch_budget_config_t;

typedef struct {
    NvU64 bytes;
    NvU64 feedback_generation;
    bool probe;
} uvm_prefetch_reservation_t;

typedef struct {
    NvU64 reserved_bytes;
    NvU64 peak_bytes;
    NvU64 window_bytes;
    NvU64 last_backoff_ns;
    NvU64 next_growth_ns;
    NvU64 feedback_generation;
    NvU64 success_ns;
    bool initialized;
    bool probe_inflight;
    bool success_pending;
    NvU64 admissions;
    NvU64 denials;
    NvU64 backoffs;
    NvU64 quick_returns;
    NvU64 capacity_stops;
    NvU64 recoveries;
    NvU64 suppressed;
    NvU64 probes;
    NvU64 probe_failures;
} uvm_prefetch_budget_t;

static inline NvU64 uvm_prefetch_budget_limit(NvU64 bytes)
{
    return bytes - bytes % UVM_PREFETCH_BUDGET_QUANTUM;
}

static inline NvU64 uvm_prefetch_budget_probe_size(NvU64 limit)
{
    return limit < 2 * UVM_PREFETCH_BUDGET_QUANTUM ? limit : 2 * UVM_PREFETCH_BUDGET_QUANTUM;
}

static inline void uvm_prefetch_budget_backoff(uvm_prefetch_budget_t *budget,
                                              const uvm_prefetch_budget_config_t *config,
                                              NvU64 now)
{
    NvU64 limit = uvm_prefetch_budget_limit(config->limit_bytes);
    NvU64 window, probe_size;

    // Invalidate ALL earlier success credit, even if reduction is rate limited.
    ++budget->feedback_generation;
    budget->success_pending = false;
    budget->next_growth_ns = now + config->recovery_ns;
    if (!limit)
        return;
    if (!budget->initialized) {
        budget->window_bytes = limit;
        budget->initialized = true;
    }
    if (!budget->window_bytes)
        return;
    if (budget->backoffs && now >= budget->last_backoff_ns &&
        now - budget->last_backoff_ns < config->backoff_ns)
        return;

    window = budget->window_bytes < limit ? budget->window_bytes : limit;
    probe_size = uvm_prefetch_budget_probe_size(limit);
    if (window <= probe_size) {
        window = 0;
    }
    else {
        window = uvm_prefetch_budget_limit(window / 2);
        if (window < probe_size)
            window = probe_size;
    }
    budget->window_bytes = window;
    budget->last_backoff_ns = now;
    ++budget->backoffs;
}

static inline NvU64 uvm_prefetch_budget_reserve(uvm_prefetch_budget_t *budget,
                                               const uvm_prefetch_budget_config_t *config,
                                               NvU64 requested, NvU64 free_bytes, NvU64 now,
                                               uvm_prefetch_reservation_t *reservation)
{
    NvU64 limit = uvm_prefetch_budget_limit(config->limit_bytes);
    NvU64 grant, headroom, ceiling;
    bool probe = false;

    reservation->bytes = 0;
    reservation->feedback_generation = 0;
    reservation->probe = false;
    if (!limit || requested < UVM_PREFETCH_BUDGET_QUANTUM)
        goto denied;
    if (!budget->initialized) {
        budget->window_bytes = limit;
        budget->initialized = true;
    }
    if (budget->window_bytes > limit)
        budget->window_bytes = limit;
    if (budget->probe_inflight)
        goto suppressed;
    if (budget->window_bytes && budget->reserved_bytes >= budget->window_bytes)
        goto denied;

    if (!budget->window_bytes) {
        // Do not replace OFF with the default window. Existing work must drain
        // before a single probe, and a passed probe must finish observation.
        if (budget->reserved_bytes || now < budget->next_growth_ns)
            goto suppressed;
        // A neighbor probe needs two slots: trigger plus one neighbor. A fixed
        // 2MiB cap or a one-block range cannot test neighbor usefulness.
        if (!budget->success_pending &&
            (limit < 2 * UVM_PREFETCH_BUDGET_QUANTUM || requested < 2 * UVM_PREFETCH_BUDGET_QUANTUM))
            goto suppressed;
    }

    // PMA can already exclude some allocations covered by a reservation.
    // Subtracting all outstanding bytes again is intentionally conservative.
    headroom = free_bytes > config->demand_reserve_bytes ? free_bytes - config->demand_reserve_bytes : 0;
    headroom = headroom > budget->reserved_bytes ? headroom - budget->reserved_bytes : 0;
    if (headroom < UVM_PREFETCH_BUDGET_QUANTUM) {
        ++budget->capacity_stops;
        uvm_prefetch_budget_backoff(budget, config, now);
        goto denied;
    }

    // Apply at most one deferred growth credit. Successful copies alone do not
    // immediately grow the window; they first need a feedback-free interval.
    if (budget->success_pending && now >= budget->next_growth_ns &&
        now >= budget->success_ns && now - budget->success_ns >= config->recovery_ns) {
        if (!budget->window_bytes) {
            budget->window_bytes = uvm_prefetch_budget_probe_size(limit);
        }
        else if (budget->window_bytes < limit) {
            NvU64 increase = limit - budget->window_bytes;
            if (increase > UVM_PREFETCH_BUDGET_QUANTUM)
                increase = UVM_PREFETCH_BUDGET_QUANTUM;
            budget->window_bytes += increase;
        }
        budget->success_pending = false;
        budget->next_growth_ns = now + config->recovery_ns;
        ++budget->recoveries;
    }

    if (!budget->window_bytes) {
        if (budget->success_pending)
            goto suppressed;
        probe = true;
        ceiling = uvm_prefetch_budget_probe_size(limit);
    }
    else {
        ceiling = budget->window_bytes;
    }
    grant = requested < ceiling ? requested : ceiling;
    if (grant > ceiling - budget->reserved_bytes)
        grant = ceiling - budget->reserved_bytes;
    if (grant > headroom)
        grant = headroom;
    grant = uvm_prefetch_budget_limit(grant);
    // Partial probes containing only the trigger provide no neighbor evidence.
    if (probe && grant < 2 * UVM_PREFETCH_BUDGET_QUANTUM) {
        ++budget->capacity_stops;
        uvm_prefetch_budget_backoff(budget, config, now);
        goto denied;
    }
    if (!grant)
        goto denied;

    budget->reserved_bytes += grant;
    if (budget->reserved_bytes > budget->peak_bytes)
        budget->peak_bytes = budget->reserved_bytes;
    if (probe) {
        budget->probe_inflight = true;
        budget->next_growth_ns = now + config->recovery_ns;
        ++budget->probes;
    }
    reservation->bytes = grant;
    reservation->feedback_generation = budget->feedback_generation;
    reservation->probe = probe;
    ++budget->admissions;
    return grant;

suppressed:
    ++budget->suppressed;
denied:
    ++budget->denials;
    return 0;
}

static inline void uvm_prefetch_budget_complete(uvm_prefetch_budget_t *budget,
                                                const uvm_prefetch_budget_config_t *config,
                                                const uvm_prefetch_reservation_t *reservation,
                                                NvU64 transferred, NvU64 neighbor_bytes,
                                                NvU64 now, bool success)
{
    bool credit;

    UVM_ASSERT(reservation->bytes <= budget->reserved_bytes);
    UVM_ASSERT(transferred <= reservation->bytes);
    UVM_ASSERT(neighbor_bytes <= transferred);
    budget->reserved_bytes -= reservation->bytes;
    credit = success && neighbor_bytes &&
             reservation->feedback_generation == budget->feedback_generation;
    if (reservation->probe) {
        UVM_ASSERT(budget->probe_inflight);
        budget->probe_inflight = false;
        // Empty, failed and invalidated probes cannot cause an immediate retry.
        budget->next_growth_ns = now + config->recovery_ns;
        if (!credit)
            ++budget->probe_failures;
    }
    if (credit && budget->window_bytes < uvm_prefetch_budget_limit(config->limit_bytes) &&
        !budget->success_pending) {
        budget->success_pending = true;
        budget->success_ns = now;
    }
}

#endif
