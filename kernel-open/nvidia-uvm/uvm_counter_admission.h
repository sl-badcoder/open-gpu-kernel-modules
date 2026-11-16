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
} uvm_counter_admission_region_t;

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
                                                NvU64 submitted, NvU64 fresh)
{
    if (!pressure) {
        r->pending = false;
        return true;
    }
    if (r->pending && r->buffer == buffer && fresh >= r->clear_serial) {
        r->pending = false;
        return true;
    }
    // Keep the earliest pending clear. Repeated pre-clear notifications must
    // neither satisfy the second observation nor postpone it indefinitely.
    if (!r->pending || r->buffer != buffer) {
        r->pending = true;
        r->buffer = buffer;
        r->clear_serial = submitted + 1;
    }
    return false;
}

#endif
