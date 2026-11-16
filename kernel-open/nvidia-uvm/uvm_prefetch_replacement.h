/* SPDX-License-Identifier: MIT */
#ifndef __UVM_PREFETCH_REPLACEMENT_H__
#define __UVM_PREFETCH_REPLACEMENT_H__

// Small, lock-free policy helper. Callers provide synchronization for a GPU's
// state and pass bytes which were actually populated (not merely requested).
// The distance is supplied by the caller because it is normally derived from
// the region's eviction/repopulation bookkeeping.

#define UVM_PREFETCH_REPLACEMENT_MIN_BATCH_BYTES (64ULL * 1024ULL * 1024ULL)

typedef struct
{
    NvU64 population_bytes;
    NvU64 trigger_population_bytes;
    NvU64 resident_bytes;
    NvU64 repopulated_bytes;
    NvU64 evicted_bytes;
    NvU64 short_quarter_bytes;
    NvU64 short_half_bytes;
    NvU64 short_budget_bytes;

    NvU64 window_population_bytes;
    NvU64 window_repopulated_bytes;
    NvU64 window_short_quarter_bytes;
    NvU64 window_short_half_bytes;
    NvU64 window_short_budget_bytes;
    NvU64 window_budget_bytes;
    NvU64 evaluations;
    NvU64 epoch_stamp;
    NvU64 epoch_roots_used;
    NvU64 epoch_roots_limit;
    NvU64 epoch_pauses;
    NvU64 last_population_bytes;
    NvU64 last_short_quarter_bytes;

    NvU32 request_stride;
} uvm_prefetch_replacement_state_t;

static inline NvU64 uvm_prefetch_replacement_add(NvU64 a, NvU64 b)
{
    NvU64 result = a + b;
    return result < a ? ~0ULL : result;
}

static inline NvU64 uvm_prefetch_replacement_batch(NvU64 budget)
{
    NvU64 scaled = budget / 16;
    return scaled > UVM_PREFETCH_REPLACEMENT_MIN_BATCH_BYTES ?
               scaled : UVM_PREFETCH_REPLACEMENT_MIN_BATCH_BYTES;
}

static inline void uvm_prefetch_replacement_init(uvm_prefetch_replacement_state_t *s)
{
    *s = (uvm_prefetch_replacement_state_t){0};
}

static inline void uvm_prefetch_replacement_evaluate(uvm_prefetch_replacement_state_t *s)
{
    NvU64 denominator = s->window_population_bytes;
    NvU64 short_bytes = s->window_short_quarter_bytes;

    s->last_population_bytes = denominator;
    s->last_short_quarter_bytes = short_bytes;
    if (denominator != 0) {
        // Avoid multiplication overflow: short/denominator >= 50%.
        if (short_bytes >= denominator - denominator / 2)
            s->request_stride = 4;
        else if (short_bytes < denominator / 4)
            s->request_stride = 1;
        else if (s->request_stride == 0)
            s->request_stride = 1;
    }
    s->evaluations = uvm_prefetch_replacement_add(s->evaluations, 1);
    s->window_population_bytes = 0;
    s->window_repopulated_bytes = 0;
    s->window_short_quarter_bytes = 0;
    s->window_short_half_bytes = 0;
    s->window_short_budget_bytes = 0;
    s->window_budget_bytes = 0;
}

// trigger=true feeds the controller. Neighbor populations still advance the
// global population total, but cannot change the trigger admission stride.
static inline void uvm_prefetch_replacement_record_population(
    uvm_prefetch_replacement_state_t *s,
    NvU64 moved_bytes,
    NvU64 repop_bytes,
    NvU64 distance_bytes,
    NvU64 residency_budget,
    bool trigger)
{
    NvU64 batch;
    s->population_bytes = uvm_prefetch_replacement_add(s->population_bytes, moved_bytes);
    s->resident_bytes = uvm_prefetch_replacement_add(s->resident_bytes, moved_bytes);
    s->repopulated_bytes = uvm_prefetch_replacement_add(s->repopulated_bytes, repop_bytes);

    if (!trigger)
        return;

    s->trigger_population_bytes =
        uvm_prefetch_replacement_add(s->trigger_population_bytes, moved_bytes);

    if (residency_budget == 0) {
        // With no meaningful budget, remain conservative and probe only
        // occasionally; no ratio is inferred from an unbounded denominator.
        s->request_stride = 4;
        return;
    }

    batch = uvm_prefetch_replacement_batch(residency_budget);
    s->window_budget_bytes = residency_budget;
    s->window_population_bytes =
        uvm_prefetch_replacement_add(s->window_population_bytes, moved_bytes);
    s->window_repopulated_bytes =
        uvm_prefetch_replacement_add(s->window_repopulated_bytes, repop_bytes);
    if (repop_bytes != 0) {
        if (distance_bytes <= residency_budget / 4) {
            s->short_quarter_bytes = uvm_prefetch_replacement_add(s->short_quarter_bytes, repop_bytes);
            s->window_short_quarter_bytes =
                uvm_prefetch_replacement_add(s->window_short_quarter_bytes, repop_bytes);
        }
        if (distance_bytes <= residency_budget / 2) {
            s->short_half_bytes = uvm_prefetch_replacement_add(s->short_half_bytes, repop_bytes);
            s->window_short_half_bytes =
                uvm_prefetch_replacement_add(s->window_short_half_bytes, repop_bytes);
        }
        if (distance_bytes <= residency_budget) {
            s->short_budget_bytes = uvm_prefetch_replacement_add(s->short_budget_bytes, repop_bytes);
            s->window_short_budget_bytes =
                uvm_prefetch_replacement_add(s->window_short_budget_bytes, repop_bytes);
        }
    }
    if (s->window_population_bytes >= batch)
        uvm_prefetch_replacement_evaluate(s);
}

static inline void uvm_prefetch_replacement_record_eviction(
    uvm_prefetch_replacement_state_t *s,
    NvU64 evicted_bytes,
    NvU64 tracked_live_bytes)
{
    s->evicted_bytes = uvm_prefetch_replacement_add(s->evicted_bytes, evicted_bytes);
    // tracked_live_bytes is the number of live bytes removed by this event.
    // Clamp it so stale/partial bookkeeping cannot underflow the gauge.
    if (tracked_live_bytes >= s->resident_bytes)
        s->resident_bytes = 0;
    else
        s->resident_bytes -= tracked_live_bytes;
}

// Bound eager reclamation in throttled mode. sampled_ns is an event/sample
// epoch supplied by the caller; repeated calls in one epoch do not replenish
// the quota. A zero timestamp intentionally grants no throttled quota.
static inline NvU64 uvm_prefetch_replacement_limit(
    uvm_prefetch_replacement_state_t *s,
    NvU64 sampled_ns,
    NvU64 requested_roots,
    NvU64 root_bytes)
{
    NvU64 quota_bytes, quota_roots, remaining;
    if (s->request_stride <= 1)
        return requested_roots;
    if (sampled_ns == 0 || root_bytes == 0) {
        s->epoch_pauses = uvm_prefetch_replacement_add(s->epoch_pauses, 1);
        return 0;
    }
    if (sampled_ns > s->epoch_stamp) {
        s->epoch_stamp = sampled_ns;
        s->epoch_roots_used = 0;
        quota_bytes = s->resident_bytes / 64;
        if (quota_bytes < root_bytes)
            quota_bytes = root_bytes;
        quota_roots = quota_bytes / root_bytes;
        if (quota_bytes % root_bytes)
            quota_roots = uvm_prefetch_replacement_add(quota_roots, 1);
        s->epoch_roots_limit = quota_roots;
    }
    if (sampled_ns != s->epoch_stamp || s->epoch_roots_used >= s->epoch_roots_limit) {
        s->epoch_pauses = uvm_prefetch_replacement_add(s->epoch_pauses, 1);
        return 0;
    }
    remaining = s->epoch_roots_limit - s->epoch_roots_used;
    return requested_roots < remaining ? requested_roots : remaining;
}

static inline void uvm_prefetch_replacement_consume(
    uvm_prefetch_replacement_state_t *s, NvU64 freed_roots)
{
    s->epoch_roots_used = uvm_prefetch_replacement_add(s->epoch_roots_used, freed_roots);
}

#endif
