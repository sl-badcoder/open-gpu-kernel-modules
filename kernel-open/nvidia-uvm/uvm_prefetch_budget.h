/* SPDX-License-Identifier: MIT */
#ifndef UVM_PREFETCH_BUDGET_H
#define UVM_PREFETCH_BUDGET_H

#include "nvtypes.h"

// All controller operations require PMM list_lock. No allocation or timer work.
#define UVM_PREFETCH_BUDGET_QUANTUM ((NvU64)2 * 1024 * 1024)
#define UVM_PREFETCH_COHORT_COUNT 4

typedef struct {
    NvU64 limit_bytes;
    NvU64 demand_reserve_bytes;
    NvU64 observation_ns;
    unsigned return_percent;
    unsigned min_blocks;
    bool backoff_enabled;
} uvm_prefetch_budget_config_t;

typedef struct {
    NvU64 bytes;
    NvU64 generation;
} uvm_prefetch_reservation_t;

typedef struct {
    NvU64 id;
    NvU64 tick;
    NvU64 blocks;
    NvU64 returns;
} uvm_prefetch_cohort_t;

typedef struct {
    NvU64 reserved_bytes;
    NvU64 peak_bytes;
    NvU64 window_bytes;
    bool initialized;
    NvU64 admissions;
    NvU64 denials;
    NvU64 backoffs;
    NvU64 quick_returns;
    NvU64 capacity_stops;
    NvU64 pressure_pauses;
    NvU64 allocation_failures;
    NvU64 prefetched_blocks;
    NvU64 mature_blocks;
    NvU64 mature_returns;
    NvU64 pending_blocks;
    NvU64 pending_returns;
    NvU64 last_blocks;
    NvU64 last_returns;
    NvU64 ratio_evaluations;
    NvU64 next_cohort_id;
    NvU64 generation;
    NvU64 discarded_blocks;
    NvU64 discarded_returns;
    NvU64 dropped_records;
    NvU64 dropped_returns;
    uvm_prefetch_cohort_t cohorts[UVM_PREFETCH_COHORT_COUNT];
} uvm_prefetch_budget_t;

static inline NvU64 uvm_prefetch_budget_limit(NvU64 bytes)
{
    return bytes - bytes % UVM_PREFETCH_BUDGET_QUANTUM;
}

static inline void uvm_prefetch_budget_backoff(uvm_prefetch_budget_t *budget,
                                              const uvm_prefetch_budget_config_t *config)
{
    NvU64 limit = uvm_prefetch_budget_limit(config->limit_bytes);
    NvU64 decrease;
    unsigned i;
    if (!config->backoff_enabled)
        return;
    if (!budget->initialized) {
        budget->window_bytes = limit;
        budget->initialized = true;
    }
    if (!budget->window_bytes)
        return;
    if (budget->window_bytes <= 2 * UVM_PREFETCH_BUDGET_QUANTUM) {
        budget->window_bytes = 0;
    }
    else {
        decrease = uvm_prefetch_budget_limit(budget->window_bytes / 4);
        if (decrease < UVM_PREFETCH_BUDGET_QUANTUM)
            decrease = UVM_PREFETCH_BUDGET_QUANTUM;
        budget->window_bytes -= decrease;
    }
    ++budget->backoffs;
    ++budget->generation;
    // Do not repeatedly shrink using populations produced at the old size.
    // IDs are never reused, including when a new cohort starts in the same tick.
    for (i = 0; i < UVM_PREFETCH_COHORT_COUNT; ++i) {
        uvm_prefetch_cohort_t *cohort = &budget->cohorts[i];
        budget->discarded_blocks += cohort->blocks;
        budget->discarded_returns += cohort->returns;
        cohort->id = 0;
        cohort->blocks = 0;
        cohort->returns = 0;
    }
}

// A cohort collects for T, then waits two complete T intervals: every entry
// gets its full prefetch->eviction->return opportunity before entering the ratio.
// Four fixed slots suffice: at most three time buckets can be immature.
// Retire lazily on events; idle GPUs need no polling or per-block traversal.
static inline void uvm_prefetch_budget_advance(uvm_prefetch_budget_t *budget,
                                              const uvm_prefetch_budget_config_t *config,
                                              NvU64 now)
{
    NvU64 tick = now / config->observation_ns;
    unsigned i;
    for (i = 0; i < UVM_PREFETCH_COHORT_COUNT; ++i) {
        uvm_prefetch_cohort_t *cohort = &budget->cohorts[i];
        if (!cohort->id || tick < cohort->tick || tick - cohort->tick < 3)
            continue;
        budget->pending_blocks += cohort->blocks;
        budget->pending_returns += cohort->returns;
        budget->mature_blocks += cohort->blocks;
        budget->mature_returns += cohort->returns;
        cohort->id = 0;
        cohort->blocks = 0;
        cohort->returns = 0;
    }
    if (budget->pending_blocks && budget->pending_blocks >= config->min_blocks) {
        NvU64 threshold;
        budget->last_blocks = budget->pending_blocks;
        budget->last_returns = budget->pending_returns;
        ++budget->ratio_evaluations;
        // ceil(blocks * percentage / 100), without overflowing the product.
        threshold = (budget->pending_blocks / 100) * config->return_percent +
                    ((budget->pending_blocks % 100) * config->return_percent + 99) / 100;
        if (budget->pending_returns >= threshold)
            uvm_prefetch_budget_backoff(budget, config);
        // Non-overlapping batches: old bad evidence cannot trigger another step.
        budget->pending_blocks = 0;
        budget->pending_returns = 0;
    }
}

static inline NvU64 uvm_prefetch_budget_record(uvm_prefetch_budget_t *budget,
                                              const uvm_prefetch_budget_config_t *config,
                                              NvU64 now, NvU64 generation)
{
    NvU64 tick = now / config->observation_ns;
    uvm_prefetch_cohort_t *cohort;
    uvm_prefetch_budget_advance(budget, config, now);
    ++budget->prefetched_blocks;
    if (generation != budget->generation) {
        ++budget->discarded_blocks;
        return 0;
    }
    cohort = &budget->cohorts[tick % UVM_PREFETCH_COHORT_COUNT];
    // A delayed timestamp must not overwrite a live, newer cohort.
    if (cohort->id && cohort->tick != tick) {
        ++budget->dropped_records;
        return 0;
    }
    if (!cohort->id) {
        cohort->id = ++budget->next_cohort_id;
        cohort->tick = tick;
    }
    ++cohort->blocks;
    return cohort->id;
}

// Caller consumes the per-block token under VA-block lock, at most once.
static inline void uvm_prefetch_budget_return(uvm_prefetch_budget_t *budget,
                                              const uvm_prefetch_budget_config_t *config,
                                              NvU64 id, NvU64 now)
{
    unsigned i;
    uvm_prefetch_budget_advance(budget, config, now);
    if (!id)
        return;
    for (i = 0; i < UVM_PREFETCH_COHORT_COUNT; ++i) {
        uvm_prefetch_cohort_t *cohort = &budget->cohorts[i];
        if (cohort->id == id && cohort->returns < cohort->blocks) {
            ++cohort->returns;
            ++budget->quick_returns;
            return;
        }
    }
    // Includes old-policy tokens discarded on a previous backoff.
    ++budget->dropped_returns;
}

static inline void uvm_prefetch_budget_pressure_pause(uvm_prefetch_budget_t *budget)
{
    ++budget->capacity_stops;
    ++budget->pressure_pauses;
}

static inline NvU64 uvm_prefetch_budget_reserve(uvm_prefetch_budget_t *budget,
                                               const uvm_prefetch_budget_config_t *config,
                                               NvU64 requested, NvU64 free_bytes, NvU64 now,
                                               uvm_prefetch_reservation_t *reservation)
{
    NvU64 limit = uvm_prefetch_budget_limit(config->limit_bytes);
    NvU64 grant, headroom, ceiling;
    reservation->bytes = 0;
    reservation->generation = 0;
    uvm_prefetch_budget_advance(budget, config, now);
    if (!limit || requested < UVM_PREFETCH_BUDGET_QUANTUM)
        goto denied;
    if (!budget->initialized) {
        budget->window_bytes = limit;
        budget->initialized = true;
    }
    if (budget->window_bytes > limit)
        budget->window_bytes = limit;
    if (!budget->window_bytes || budget->reserved_bytes >= budget->window_bytes)
        goto denied;
    headroom = free_bytes > config->demand_reserve_bytes ? free_bytes - config->demand_reserve_bytes : 0;
    headroom = headroom > budget->reserved_bytes ? headroom - budget->reserved_bytes : 0;
    if (headroom < UVM_PREFETCH_BUDGET_QUANTUM) {
        uvm_prefetch_budget_pressure_pause(budget);
        goto denied;
    }
    ceiling = budget->window_bytes - budget->reserved_bytes;
    grant = requested < ceiling ? requested : ceiling;
    if (grant > headroom)
        grant = headroom;
    grant = uvm_prefetch_budget_limit(grant);
    if (!grant)
        goto denied;
    budget->reserved_bytes += grant;
    if (budget->reserved_bytes > budget->peak_bytes)
        budget->peak_bytes = budget->reserved_bytes;
    reservation->bytes = grant;
    reservation->generation = budget->generation;
    ++budget->admissions;
    return grant;
denied:
    ++budget->denials;
    return 0;
}

static inline void uvm_prefetch_budget_complete(uvm_prefetch_budget_t *budget,
                                                const uvm_prefetch_reservation_t *reservation,
                                                NvU64 transferred)
{
    UVM_ASSERT(reservation->bytes <= budget->reserved_bytes);
    UVM_ASSERT(transferred <= reservation->bytes);
    budget->reserved_bytes -= reservation->bytes;
}

#endif
