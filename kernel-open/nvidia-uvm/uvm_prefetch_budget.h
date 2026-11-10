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
    NvU64 owner;
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
    NvU64 success_owner;
    NvU64 sequential_owner;
    NvU64 sequential_address;
    NvU64 sequential_ns;
    NvU64 sequential_start_ns;
    NvU64 feedback_first_ns;
    NvU64 feedback_last_ns;
    unsigned feedback_samples;
    bool feedback_quick_return;
    unsigned sequential_blocks;
    bool sequential_reverse;
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
    NvU64 sequential_recoveries;
    NvU64 pressure_pauses;
    NvU64 allocation_failures;
    NvU64 feedback_confirmations;
} uvm_prefetch_budget_t;

static inline NvU64 uvm_prefetch_budget_limit(NvU64 bytes)
{
    return bytes - bytes % UVM_PREFETCH_BUDGET_QUANTUM;
}

static inline NvU64 uvm_prefetch_budget_probe_size(NvU64 limit)
{
    return limit < 2 * UVM_PREFETCH_BUDGET_QUANTUM ? limit : 2 * UVM_PREFETCH_BUDGET_QUANTUM;
}

// Observe demand signals only, never the speculative walk. The owner is an
// opaque range identity, not a retained pointer. One stream per GPU keeps this
// conservative when unrelated allocations interleave. Notification sorting
// means this detects spatial locality, not the order of GPU instructions.
static inline void uvm_prefetch_budget_observe(uvm_prefetch_budget_t *budget,
                                              const uvm_prefetch_budget_config_t *config,
                                              NvU64 owner, NvU64 address, NvU64 now)
{
    bool reverse = address < budget->sequential_address;
    NvU64 distance = reverse ? budget->sequential_address - address : address - budget->sequential_address;

    if (budget->sequential_blocks && budget->sequential_owner == owner &&
        now >= budget->sequential_ns && now >= budget->sequential_start_ns &&
        now - budget->sequential_start_ns <= config->recovery_ns) {
        // Repeated signals for one block cannot manufacture a run or refresh it.
        if (!distance)
            return;
        if (distance == UVM_PREFETCH_BUDGET_QUANTUM &&
            (budget->sequential_blocks == 1 || reverse == budget->sequential_reverse)) {
            if (budget->sequential_blocks < 4)
                ++budget->sequential_blocks;
            budget->sequential_reverse = reverse;
        }
        else {
            budget->sequential_blocks = 1;
            budget->sequential_start_ns = now;
        }
    }
    else {
        budget->sequential_blocks = 1;
        budget->sequential_start_ns = now;
    }
    budget->sequential_owner = owner;
    budget->sequential_address = address;
    budget->sequential_ns = now;
}

static inline void uvm_prefetch_budget_invalidate(uvm_prefetch_budget_t *budget)
{
    ++budget->feedback_generation;
    budget->success_pending = false;
    budget->sequential_blocks = 0;
}

// Apply a confirmed reduction. Admission pressure alone must not call this.
static inline void uvm_prefetch_budget_backoff(uvm_prefetch_budget_t *budget,
                                              const uvm_prefetch_budget_config_t *config,
                                              NvU64 now)
{
    NvU64 limit = uvm_prefetch_budget_limit(config->limit_bytes);
    NvU64 window, probe_size;

    // Invalidate ALL earlier success credit, even if reduction is rate limited.
    uvm_prefetch_budget_invalidate(budget);
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
        NvU64 decrease = uvm_prefetch_budget_limit(window / 4);
        if (decrease < UVM_PREFETCH_BUDGET_QUANTUM)
            decrease = UVM_PREFETCH_BUDGET_QUANTUM;
        window -= decrease;
        if (window < probe_size)
            window = probe_size;
    }
    budget->window_bytes = window;
    budget->last_backoff_ns = now;
    ++budget->backoffs;
}

static inline void uvm_prefetch_budget_pressure_pause(uvm_prefetch_budget_t *budget)
{
    ++budget->capacity_stops;
    ++budget->pressure_pauses;
}

// Only genuine no-eviction allocation failures and consumed quick-return
// markers belong here. A callback burst is not sustained negative evidence.
static inline void uvm_prefetch_budget_feedback(uvm_prefetch_budget_t *budget,
                                               const uvm_prefetch_budget_config_t *config,
                                               bool quick_return, NvU64 now)
{
    NvU64 horizon = config->recovery_ns;
    if (horizon < 2 * config->backoff_ns)
        horizon = 2 * config->backoff_ns;

    if (quick_return) {
        ++budget->quick_returns;
    }
    else {
        ++budget->capacity_stops;
        ++budget->allocation_failures;
    }

    if (!budget->feedback_samples || budget->feedback_quick_return != quick_return ||
        now < budget->feedback_last_ns || now - budget->feedback_first_ns > horizon) {
        budget->feedback_samples = 1;
        budget->feedback_first_ns = now;
        budget->feedback_last_ns = now;
        budget->feedback_quick_return = quick_return;
    }
    else if (now - budget->feedback_last_ns >= config->backoff_ns) {
        ++budget->feedback_samples;
        budget->feedback_last_ns = now;
        if (budget->feedback_samples == 3) {
            budget->feedback_samples = 0;
            ++budget->feedback_confirmations;
            uvm_prefetch_budget_backoff(budget, config, now);
            return;
        }
    }

    // Invalidate old copy credit even for an unconfirmed real failure, but
    // do not continually push the recovery deadline into the future.
    uvm_prefetch_budget_invalidate(budget);
}

static inline NvU64 uvm_prefetch_budget_reserve(uvm_prefetch_budget_t *budget,
                                               const uvm_prefetch_budget_config_t *config,
                                               NvU64 owner, NvU64 requested, NvU64 free_bytes, NvU64 now,
                                               uvm_prefetch_reservation_t *reservation)
{
    NvU64 limit = uvm_prefetch_budget_limit(config->limit_bytes);
    NvU64 grant, headroom, ceiling, observation_ns;
    bool sequential;
    bool probe = false;

    reservation->bytes = 0;
    reservation->feedback_generation = 0;
    reservation->owner = owner;
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
        uvm_prefetch_budget_pressure_pause(budget);
        goto denied;
    }

    sequential = budget->window_bytes && budget->sequential_blocks >= 4 &&
                 budget->sequential_owner == budget->success_owner &&
                 now >= budget->sequential_ns && now - budget->sequential_ns <= config->recovery_ns;
    observation_ns = sequential ? config->backoff_ns : config->recovery_ns;

    // Apply at most one deferred growth credit. Successful copies alone do not
    // immediately grow the window; they first need a feedback-free interval.
    if (budget->success_pending && now >= budget->next_growth_ns &&
        now >= budget->success_ns && now - budget->success_ns >= observation_ns) {
        if (!budget->window_bytes) {
            budget->window_bytes = uvm_prefetch_budget_probe_size(limit);
        }
        else if (budget->window_bytes < limit) {
            NvU64 increase = limit - budget->window_bytes;
            NvU64 step = sequential ? budget->window_bytes : UVM_PREFETCH_BUDGET_QUANTUM;
            if (increase > step)
                increase = step;
            budget->window_bytes += increase;
            if (sequential) {
                ++budget->sequential_recoveries;
                budget->sequential_blocks = 0;
            }
        }
        budget->success_pending = false;
        budget->next_growth_ns = now + observation_ns;
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
        uvm_prefetch_budget_pressure_pause(budget);
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
    // A successful neighbor attempt breaks an allocation-failure streak.
    // Copy completion cannot disprove churn, so keep quick-return evidence.
    if (credit && !budget->feedback_quick_return)
        budget->feedback_samples = 0;
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
        budget->success_owner = reservation->owner;
    }
}

#endif
