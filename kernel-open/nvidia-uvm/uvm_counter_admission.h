/* SPDX-License-Identifier: MIT */
#ifndef __UVM_COUNTER_ADMISSION_H__
#define __UVM_COUNTER_ADMISSION_H__

// Pure policy. Buffer state is protected by its ISR service lock; region state
// follows the virtual block/GPU lifetime and is protected by the VA-block lock.
typedef struct
{
    NvU64 submitted;
    NvU64 fresh;
    NvU64 fence;
    NvU64 batch_fresh;
    NvU32 remaining;
    bool fencing;
    bool tracking;
} uvm_counter_admission_buffer_t;

typedef struct
{
    NvU64 clear_serial;
    NvU32 buffer;
    bool pending;
    // Fits in the existing padding; counts distinct clear-separated reports.
    unsigned char observations;
    unsigned char required;
} uvm_counter_admission_region_t;

// One complete-eviction episode per logical block/GPU. Distance is CPU-policy
// population bytes on this GPU, not elapsed time or PCIe wire traffic.
typedef struct
{
    NvU64 population_at_eviction;
    bool evicted;
} uvm_counter_admission_history_t;

static inline NvU32 uvm_counter_admission_required(uvm_counter_admission_history_t *h,
                                                  NvU64 population, NvU64 budget,
                                                  NvU32 base, NvU32 extra)
{
    bool recent = h->evicted && population - h->population_at_eviction < budget;
    if (base < 2)
        base = 2;
    if (base > 8)
        base = 8;
    if (extra > 8 - base)
        extra = 8 - base;
    return base + (recent ? extra : 0);
}

// Called before fetching, after testing clear completion and reading PUT.
// Capture AFTER completion, so already queued notifications cannot be evidence
// of reaccess after the clear. A batch crossing the fence stays conservative:
// the new serial becomes usable only by the following batch.
static inline void uvm_counter_admission_prepare(uvm_counter_admission_buffer_t *s,
                                                 bool completed, NvU32 queued)
{
    if (!s->fencing && completed && s->submitted > s->fresh) {
        s->fence = s->submitted;
        s->remaining = queued;
        s->fencing = queued != 0;
        if (!queued)
            s->fresh = s->fence;
    }
    s->batch_fresh = s->fresh;
}

static inline void uvm_counter_admission_consumed(uvm_counter_admission_buffer_t *s, NvU32 count)
{
    if (s->fencing) {
        if (count >= s->remaining) {
            s->fresh = s->fence;
            s->remaining = 0;
            s->fencing = false;
        }
        else {
            s->remaining -= count;
        }
    }
}

static inline bool uvm_counter_admission_decide(uvm_counter_admission_region_t *r,
                                                bool pressure, NvU32 buffer,
                                                NvU64 submitted, NvU64 fresh,
                                                NvU32 required_observations)
{
    if (!pressure) {
        r->pending = false;
        r->observations = 0;
        return true;
    }
    if (required_observations < 2)
        required_observations = 2;
    if (required_observations > 8)
        required_observations = 8;
    if (r->pending && r->buffer == buffer && fresh >= r->clear_serial) {
        if (++r->observations >= r->required) {
            r->pending = false;
            r->observations = 0;
            return true;
        }
        // A further observation must follow this report's own clear. Multiple
        // reports against the same fresh boundary never advance the count.
        r->clear_serial = submitted + 1;
        return false;
    }
    // Keep the earliest pending clear. Repeated pre-clear notifications must
    // neither satisfy the second observation nor postpone it indefinitely.
    if (!r->pending || r->buffer != buffer) {
        r->pending = true;
        r->buffer = buffer;
        r->clear_serial = submitted + 1;
        r->observations = 1;
        r->required = required_observations;
    }
    return false;
}

#endif
