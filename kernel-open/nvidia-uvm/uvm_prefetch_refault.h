/* SPDX-License-Identifier: MIT */
#ifndef UVM_PREFETCH_REFAULT_H
#define UVM_PREFETCH_REFAULT_H

#include "nvtypes.h"

// All calls require the owning PMM list_lock. No pointers, allocation or timers.
#define UVM_REFAULT_TILES 256u
#define UVM_REFAULT_BLOCK_SHIFT 21u
#define UVM_REFAULT_TILE_SHIFT 26u
#define UVM_REFAULT_MAX_WINDOW 31u

typedef struct {
    NvU64 capacity_bytes;
    unsigned short_distance_percent;
    unsigned max_neighbors;
} uvm_prefetch_refault_config_t;

// Embedded in the owning VA-block GPU state, protected by that block's lock.
typedef struct {
    NvU64 evict_clock;
    NvU64 last_return_clock;
    bool evicted;
    unsigned short_returns;
    bool has_observation;
    NvU64 last_batch;
    unsigned last_buffer;
} uvm_prefetch_refault_block_t;

typedef struct {
    bool valid;
    NvU64 range_id;
    NvU64 tile;
    NvU32 seen;
    unsigned window;
    unsigned credits;
    NvU32 populated;
    NvU32 returned;
    NvU32 short_returned;
    NvU64 epoch_start;
    NvU64 last_demand;
} uvm_prefetch_refault_tile_t;

typedef struct {
    NvU64 demand_bytes;
    NvU64 evictions;
    NvU64 refaults;
    NvU64 short_refaults;
    NvU64 last_distance;
    NvU64 replacements;
    NvU64 plans;
    NvU64 paused;
    NvU64 growths;
    NvU64 reductions;
    uvm_prefetch_refault_tile_t tiles[UVM_REFAULT_TILES];
} uvm_prefetch_refault_t;

static inline NvU64 uvm_prefetch_refault_hash(NvU64 range, NvU64 value)
{
    NvU64 x = range ^ (value + 0x9e3779b97f4a7c15ULL + (range << 6) + (range >> 2));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uvm_prefetch_refault_tile_t *
uvm_prefetch_refault_tile(uvm_prefetch_refault_t *t, NvU64 range, NvU64 tile)
{
    unsigned i = (unsigned)(uvm_prefetch_refault_hash(range, tile ^ 0x51) % UVM_REFAULT_TILES);
    uvm_prefetch_refault_tile_t *e = &t->tiles[i];
    if (!e->valid || e->range_id != range || e->tile != tile) {
        if (e->valid)
            ++t->replacements;
        *e = (uvm_prefetch_refault_tile_t){.valid = true, .range_id = range, .tile = tile,
                                         .window = 1, .epoch_start = t->demand_bytes,
                                         .last_demand = t->demand_bytes};
    }
    return e;
}

static inline NvU64 uvm_prefetch_refault_horizon(const uvm_prefetch_refault_config_t *c)
{
    NvU64 quarter = c->capacity_bytes / 4;
    return quarter < (2ULL << 20) ? (2ULL << 20) : quarter;
}

static inline void uvm_prefetch_refault_age(uvm_prefetch_refault_t *t,
                                            const uvm_prefetch_refault_config_t *c,
                                            uvm_prefetch_refault_tile_t *s)
{
    NvU64 horizon = uvm_prefetch_refault_horizon(c);
    if (t->demand_bytes - s->last_demand >= c->capacity_bytes) {
        s->window = 1;
        s->seen = 0;
        s->populated = 0;
        s->returned = 0;
        s->short_returned = 0;
        s->credits = 0;
        s->epoch_start = t->demand_bytes;
    }
    else if (t->demand_bytes - s->epoch_start >= horizon) {
        s->seen = 0;
        s->populated = 0;
        s->returned = 0;
        s->short_returned = 0;
        s->credits = 0;
        if (!s->window)
            s->window = 1;
        s->epoch_start = t->demand_bytes;
    }
}

static inline void uvm_prefetch_refault_evict(uvm_prefetch_refault_t *t,
                                              uvm_prefetch_refault_block_t *e)
{
    // Several partial evictions belong to the same episode. Preserve its start.
    if (!e->evicted) {
        e->evict_clock = t->demand_bytes;
        e->evicted = true;
    }
    ++t->evictions;
}

// Successful speculative residency is spatial coverage only, never demand or
// proof of use. A later distinct remote demand must earn any growth credit.
static inline void uvm_prefetch_refault_population(uvm_prefetch_refault_t *t,
                                                  const uvm_prefetch_refault_config_t *c,
                                                  NvU64 range, NvU64 address)
{
    uvm_prefetch_refault_tile_t *s = uvm_prefetch_refault_tile(t, range, address >> 26);
    uvm_prefetch_refault_age(t, c, s);
    s->populated |= (NvU32)1 << ((address >> 21) & 31);
}

static inline bool uvm_prefetch_refault_short(const uvm_prefetch_refault_config_t *c,
                                              NvU64 distance)
{
    unsigned percent = c->short_distance_percent > 100 ? 100 : c->short_distance_percent;
    NvU64 limit = (c->capacity_bytes / 100) * percent +
                  ((c->capacity_bytes % 100) * percent) / 100;
    return c->capacity_bytes && distance <= limit;
}

static inline void uvm_prefetch_refault_demand(uvm_prefetch_refault_t *t,
                                               const uvm_prefetch_refault_config_t *c,
                                               uvm_prefetch_refault_block_t *e,
                                               NvU64 range, NvU64 address,
                                               NvU64 bytes, unsigned buffer,
                                               NvU64 batch, bool valid, bool returned_page)
{
    uvm_prefetch_refault_tile_t *s;
    NvU64 block = address >> 21;
    NvU32 bit = (NvU32)1 << (block & 31);
    NvU64 distance = 0;
    bool is_return, is_short = false;
    bool fresh;

    // The return distance excludes this demand's own transfer. Failed/partial
    // service still advances progress, without training or consuming history.
    if (!valid || !c->capacity_bytes || !e) {
        t->demand_bytes += bytes;
        return;
    }
    s = uvm_prefetch_refault_tile(t, range, address >> 26);
    uvm_prefetch_refault_age(t, c, s);
    fresh = !e->has_observation || e->last_batch != batch || e->last_buffer != buffer;
    is_return = fresh && e->evicted && returned_page;
    if (is_return)
        distance = t->demand_bytes - e->evict_clock;

    if (fresh) {
        e->has_observation = true;
        e->last_batch = batch;
        e->last_buffer = buffer;
        s->last_demand = t->demand_bytes;
    }
    if (is_return) {
        e->evicted = false;
        is_short = uvm_prefetch_refault_short(c, distance);
        if (t->demand_bytes - e->last_return_clock >= c->capacity_bytes)
            e->short_returns = 0;
        e->last_return_clock = t->demand_bytes;
        e->short_returns = is_short ? (e->short_returns < 2 ? e->short_returns + 1 : 2) : 0;
        ++t->refaults;
        t->last_distance = distance;
        s->returned |= bit;
        if (is_short)
            s->short_returned |= bit;
        else
            s->short_returned &= ~bit;
        if (is_short) {
            unsigned next = e->short_returns >= 2 ? 0 : (s->window > 1 ? s->window / 2 : 1);
            ++t->short_refaults;
            if (next < s->window)
                ++t->reductions;
            s->credits = 0;
            s->window = next;
        }
        else if (!s->window) {
            s->window = 1;
        }
    }
    if (fresh && !(s->seen & bit) && !is_short) {
        unsigned pos = (unsigned)(block & 31);
        NvU32 covered = s->seen | s->populated;
        bool adjacent = (pos && (covered & (bit >> 1))) ||
                        (pos < 31 && (covered & (bit << 1)));
        // Spatial density, independent of notification sort direction. Repeated
        // demand for the same offset cannot earn additional growth credits.
        if (adjacent && s->window) {
            if (++s->credits >= 4) {
                unsigned cap = c->max_neighbors < UVM_REFAULT_MAX_WINDOW ?
                                   c->max_neighbors : UVM_REFAULT_MAX_WINDOW;
                if (s->window < cap) {
                    s->window = s->window > cap / 2 ? cap : s->window * 2;
                    ++t->growths;
                }
                s->credits = 0;
            }
        }
        else if (s->credits) {
            --s->credits;
        }
    }
    if (fresh)
        s->seen |= bit;
    t->demand_bytes += bytes;
}

static inline unsigned uvm_prefetch_refault_neighbors(uvm_prefetch_refault_t *t,
                                                      const uvm_prefetch_refault_config_t *c,
                                                      NvU64 range, NvU64 address)
{
    uvm_prefetch_refault_tile_t *s;
    ++t->plans;
    if (!c->capacity_bytes || !c->max_neighbors) {
        ++t->paused;
        return 0;
    }
    s = uvm_prefetch_refault_tile(t, range, address >> 26);
    uvm_prefetch_refault_age(t, c, s);
    {
        unsigned returns = 0, shorts = 0;
        NvU32 bits;
        for (bits = s->returned; bits; bits &= bits - 1)
            ++returns;
        for (bits = s->short_returned; bits; bits &= bits - 1)
            ++shorts;
        // Pressure is local and requires distinct returning blocks. One hot
        // block cannot throttle unrelated allocations across the GPU.
        if (returns >= 4 && shorts * 2 >= returns) {
            ++t->paused;
            return 0;
        }
    }
    if (!s->window) {
        ++t->paused;
        return 0;
    }
    return s->window < c->max_neighbors ? s->window : c->max_neighbors;
}
#endif
