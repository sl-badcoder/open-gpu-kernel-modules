/* SPDX-License-Identifier: MIT */
// Positive co-demand evidence only; absence is never scored as a prefetch miss.
// All operations are serialized by the owning GPU PMM list_lock.
#ifndef __UVM_PREFETCH_PATTERN_H__
#define __UVM_PREFETCH_PATTERN_H__

#include "nvtypes.h"

#define UVM_PREFETCH_PATTERN_SLOTS 512u
#define UVM_PREFETCH_PATTERN_REGION_SHIFT 22u
#define UVM_PREFETCH_PATTERN_TILE_SHIFT 26u
#define UVM_PREFETCH_PATTERN_EXPIRY_EVENTS 512ull

typedef struct {
    bool valid;
    NvU64 range_id;
    NvU64 tile;
    unsigned buffer_index;
    NvU64 batch_id;
    NvU16 current;
    NvU16 previous;
    NvU16 older;
    NvU16 learned;
    NvU16 active;
    NvU64 last_learned_event;
} uvm_prefetch_pattern_entry_t;

typedef struct {
    uvm_prefetch_pattern_entry_t entries[UVM_PREFETCH_PATTERN_SLOTS];
    NvU64 events;
    NvU64 confirmations;
    NvU64 replacements;
    NvU64 predictions;
    NvU64 learned_predictions;
    NvU64 expirations;
} uvm_prefetch_pattern_t;

static inline unsigned uvm_prefetch_pattern_popcount16(NvU16 value)
{
    unsigned count = 0;
    while (value) {
        value &= (NvU16)(value - 1u);
        ++count;
    }
    return count;
}

static inline unsigned uvm_prefetch_pattern_hash(NvU64 range_id,
                                                  NvU64 tile,
                                                  unsigned buffer_index)
{
    NvU64 x = range_id ^ (tile * 0x9e3779b97f4a7c15ull) ^
              ((NvU64)buffer_index * 0xbf58476d1ce4e5b9ull);
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return (unsigned)x & (UVM_PREFETCH_PATTERN_SLOTS - 1u);
}

static inline void uvm_prefetch_pattern_reset_history(uvm_prefetch_pattern_entry_t *entry)
{
    entry->current = 0;
    entry->previous = 0;
    entry->older = 0;
    entry->learned = 0;
    entry->active = 0;
    entry->last_learned_event = 0;
}

static inline uvm_prefetch_pattern_entry_t *
uvm_prefetch_pattern_find(uvm_prefetch_pattern_t *table,
                          NvU64 range_id,
                          NvU64 tile,
                          unsigned buffer_index)
{
    unsigned slot = uvm_prefetch_pattern_hash(range_id, tile, buffer_index);
    uvm_prefetch_pattern_entry_t *entry = &table->entries[slot];
    if (!entry->valid || entry->range_id != range_id || entry->tile != tile ||
        entry->buffer_index != buffer_index)
        return NULL;
    return entry;
}

static inline uvm_prefetch_pattern_entry_t *
uvm_prefetch_pattern_get(uvm_prefetch_pattern_t *table,
                         NvU64 range_id,
                         NvU64 tile,
                         unsigned buffer_index)
{
    unsigned slot = uvm_prefetch_pattern_hash(range_id, tile, buffer_index);
    uvm_prefetch_pattern_entry_t *entry = &table->entries[slot];
    if (entry->valid && (entry->range_id != range_id || entry->tile != tile ||
                         entry->buffer_index != buffer_index)) {
        ++table->replacements;
        entry->valid = false;
    }
    if (!entry->valid) {
        entry->valid = true;
        entry->range_id = range_id;
        entry->tile = tile;
        entry->buffer_index = buffer_index;
        entry->batch_id = 0;
        uvm_prefetch_pattern_reset_history(entry);
    }
    return entry;
}

static inline void
uvm_prefetch_pattern_maybe_expire(uvm_prefetch_pattern_t *table,
                                  uvm_prefetch_pattern_entry_t *entry)
{
    if (entry->learned && table->events - entry->last_learned_event >=
                              UVM_PREFETCH_PATTERN_EXPIRY_EVENTS) {
        ++table->expirations;
        uvm_prefetch_pattern_reset_history(entry);
    }
}

static inline bool
uvm_prefetch_pattern_observe(uvm_prefetch_pattern_t *table,
                             NvU64 range_id,
                             NvU64 address,
                             unsigned buffer_index,
                             NvU64 batch_id,
                             bool *confirmed)
{
    NvU64 region = address >> UVM_PREFETCH_PATTERN_REGION_SHIFT;
    NvU64 tile = address >> UVM_PREFETCH_PATTERN_TILE_SHIFT;
    NvU16 bit = (NvU16)(1u << (region & 15u));
    uvm_prefetch_pattern_entry_t *entry =
        uvm_prefetch_pattern_get(table, range_id, tile, buffer_index);
    NvU16 stable;

    if (confirmed)
        *confirmed = false;
    if (entry->batch_id == batch_id && (entry->current & bit))
        return false;
    ++table->events;
    uvm_prefetch_pattern_maybe_expire(table, entry);
    if (entry->batch_id != batch_id) {
        // Replace a group only with repeated positive evidence from complete
        // batches. Do not union unrelated workload phases indefinitely.
        stable = entry->current & entry->previous & entry->older;
        if (uvm_prefetch_pattern_popcount16(stable) >= 2u) {
            entry->learned = stable;
            entry->last_learned_event = table->events;
        }
        entry->active = entry->learned;
        entry->older = entry->previous;
        entry->previous = entry->current;
        entry->current = 0;
        entry->batch_id = batch_id;
    }
    entry->current |= bit;
    stable = entry->current & entry->previous & entry->older;
    if (confirmed)
        *confirmed = (uvm_prefetch_pattern_popcount16(stable) >= 2u) &&
                    ((stable & bit) != 0);
    if (confirmed && *confirmed)
        ++table->confirmations;
    return true;
}

static inline NvU16
uvm_prefetch_pattern_predict(uvm_prefetch_pattern_t *table,
                             NvU64 range_id,
                             NvU64 address,
                             unsigned buffer_index,
                             NvU64 batch_id)
{
    NvU64 region = address >> UVM_PREFETCH_PATTERN_REGION_SHIFT;
    NvU64 tile = address >> UVM_PREFETCH_PATTERN_TILE_SHIFT;
    NvU16 bit = (NvU16)(1u << (region & 15u));
    uvm_prefetch_pattern_entry_t *entry =
        uvm_prefetch_pattern_find(table, range_id, tile, buffer_index);
    NvU16 result = bit;
    ++table->predictions;
    if (!entry)
        return result;
    uvm_prefetch_pattern_maybe_expire(table, entry);
    if (entry->batch_id == batch_id && (entry->active & bit)) {
        result = entry->active & (NvU16)~entry->current;
        result |= bit;
    }
    return result;
}

#endif
