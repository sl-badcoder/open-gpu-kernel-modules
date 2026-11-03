// SPDX-License-Identifier: MIT
/*******************************************************************************
 * Helpers for consuming the legacy vidmem access-bit bitmap.
 *
 * The legacy RM interface does not expose range starts. Callers must provide
 * the start of the configured linear window explicitly. Zero means cold only
 * when every bit covering the requested physical interval is enabled.
 ******************************************************************************/

#ifndef __UVM_VAB_H__
#define __UVM_VAB_H__

#include "nvtypes.h"

static inline bool uvm_vab_region_accessed(const NvU64 *bits,
                                           const NvU64 *enabled_mask,
                                           NvU32 word_count,
                                           NvU32 granularity,
                                           NvU64 window_base,
                                           NvU64 address,
                                           NvU64 size)
{
    NvU64 bit_size;
    NvU64 end;
    NvU64 first;
    NvU64 last;
    NvU64 bit;

    if (!bits || !enabled_mask || word_count == 0 || size == 0 || granularity > 15)
        return false;

    // 64 KiB << 15 is 2 GiB and is representable without shifting overflow.
    bit_size = (NvU64)64 * 1024 << granularity;
    if (address < window_base || size - 1 > ~address)
        return false;

    end = address + size - 1;
    if (end < window_base)
        return false;

    first = (address - window_base) / bit_size;
    last = (end - window_base) / bit_size;
    if (first >= (NvU64)word_count * 64)
        return false;
    if (last >= (NvU64)word_count * 64)
        last = (NvU64)word_count * 64 - 1;

    for (bit = first; bit <= last; ++bit) {
        NvU32 word = (NvU32)(bit / 64);
        NvU64 mask = (NvU64)1 << (bit % 64);
        if ((enabled_mask[word] & mask) && (bits[word] & mask))
            return true;
        if (bit == last)
            break;
    }

    return false;
}

static inline bool uvm_vab_region_cold(const NvU64 *bits,
                                       const NvU64 *enabled_mask,
                                       NvU32 word_count,
                                       NvU32 granularity,
                                       NvU64 window_base,
                                       NvU64 address,
                                       NvU64 size)
{
    NvU64 bit_size;
    NvU64 first;
    NvU64 last;
    NvU64 bit;

    if (!bits || !enabled_mask || !word_count || !size || granularity > 15 ||
        address < window_base || size - 1 > ~address)
        return false;

    bit_size = (NvU64)64 * 1024 << granularity;
    first = (address - window_base) / bit_size;
    last = (address + size - 1 - window_base) / bit_size;
    if (last >= (NvU64)word_count * 64)
        return false;

    for (bit = first; bit <= last; ++bit) {
        if (!(enabled_mask[bit / 64] & ((NvU64)1 << (bit % 64))))
            return false;
    }

    return !uvm_vab_region_accessed(bits, enabled_mask, word_count, granularity, window_base, address, size);
}

static inline bool uvm_vab_sample_fresh(NvU64 now, NvU64 sampled, NvU64 max_age)
{
    return sampled != 0 && now >= sampled && now - sampled <= max_age;
}

#endif
