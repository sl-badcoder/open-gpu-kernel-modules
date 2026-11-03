/* SPDX-License-Identifier: MIT */
#ifndef UVM_VAB_H
#define UVM_VAB_H

#include "nvtypes.h"

#define UVM_VAB_WORD_COUNT 64

typedef enum {
    UVM_VAB_REGION_UNKNOWN = 0,
    UVM_VAB_REGION_COLD = 1,
    UVM_VAB_REGION_HOT = 2,
} uvm_vab_region_state_t;

typedef struct {
    NvU64 access_bits[UVM_VAB_WORD_COUNT];
    NvU64 enabled_mask[UVM_VAB_WORD_COUNT];
    NvU64 sampled_ns;
    NvU64 linear_base;
    NvU32 granularity;
    bool valid;
    bool eviction_enabled;
} uvm_vab_snapshot_t;

static inline bool uvm_vab_sample_fresh(NvU64 now, NvU64 sampled, NvU64 max_age)
{
    return sampled != 0 && now >= sampled && now - sampled <= max_age;
}

// Classify bits at sample time. Observation passes legacy migration history;
// eviction passes the dedicated completion/readiness boundary after validating
// the root's ownership generation. Neither mode guesses hardware reset timing.
static inline uvm_vab_region_state_t uvm_vab_classify(const uvm_vab_snapshot_t *snapshot,
                                                       NvU64 address,
                                                       NvU64 size,
                                                       NvU64 resident_since_ns)
{
    NvU64 max_address = ~(NvU64)0;
    NvU64 bit_size;
    NvU64 end;
    NvU64 first_bit;
    NvU64 last_bit;
    NvU64 bit;
    bool hot = false;

    if (snapshot == (void *)0 || !snapshot->valid ||
        snapshot->linear_base == max_address || snapshot->granularity > 15 ||
        snapshot->sampled_ns == 0 || snapshot->sampled_ns < resident_since_ns ||
        size == 0 || address > max_address - (size - 1) ||
        address < snapshot->linear_base)
        return UVM_VAB_REGION_UNKNOWN;

    bit_size = ((NvU64)64 * 1024) << snapshot->granularity;
    end = address + size - 1;
    if (end < snapshot->linear_base)
        return UVM_VAB_REGION_UNKNOWN;

    first_bit = (address - snapshot->linear_base) / bit_size;
    last_bit = (end - snapshot->linear_base) / bit_size;
    if (last_bit >= (NvU64)UVM_VAB_WORD_COUNT * 64)
        return UVM_VAB_REGION_UNKNOWN;

    for (bit = first_bit;; ++bit) {
        NvU32 word = (NvU32)(bit / 64);
        NvU64 mask = (NvU64)1 << (bit % 64);

        /* A disabled bit means the hardware did not provide coverage. */
        if ((snapshot->enabled_mask[word] & mask) == 0)
            return UVM_VAB_REGION_UNKNOWN;
        if (snapshot->access_bits[word] & mask)
            hot = true;
        if (bit == last_bit)
            break;
    }

    return hot ? UVM_VAB_REGION_HOT : UVM_VAB_REGION_COLD;
}

#endif
