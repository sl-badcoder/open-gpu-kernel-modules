// SPDX-License-Identifier: MIT
// Native-granularity VAB observation alongside the demand-first policy.
#include "uvm_cpu_block_policy.h"
#include "uvm_api.h"
#include "uvm_gpu.h"
#include "uvm_global.h"
#include "uvm_procfs.h"
#include "nv_uvm_interface.h"
#include <linux/bitops.h>
#include <linux/string.h>

// Observation is enabled independently of the unverified physical mapping.
// 0 = baseline (no dumps), 1 = observe, 2 = cold-first, 3 = VAB recency.
static unsigned uvm_perf_vab_mode = 3;
static unsigned uvm_perf_vab_interval_ms = 100;
static unsigned long long uvm_perf_vab_eviction_linear_base = ~0ULL;
module_param(uvm_perf_vab_mode, uint, S_IRUGO | S_IWUSR);
module_param(uvm_perf_vab_interval_ms, uint, S_IRUGO | S_IWUSR);
module_param(uvm_perf_vab_eviction_linear_base, ullong, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_vab_mode, "VAB mode: 0=no sampling, 1=observe, 2=cold-first eviction, 3=VAB recency eviction (default); other values disable sampling.");
MODULE_PARM_DESC(uvm_perf_vab_interval_ms, "VAB dump interval in ms (default 100, clamped to 5..1000).");
MODULE_PARM_DESC(uvm_perf_vab_eviction_linear_base, "Explicit assertion of a single linear physical VAB bit mapping; unset leaves region classification unknown.");

static unsigned vab_mode(void)
{
    unsigned mode = READ_ONCE(uvm_perf_vab_mode);
    return mode <= 3 ? mode : 0;
}

static unsigned vab_interval_ms(void)
{
    return clamp(READ_ONCE(uvm_perf_vab_interval_ms), 5U, 1000U);
}

// cache_lock must be held. Mode and freshness are applied at consumption time,
// so switching to baseline immediately invalidates observations.
static void vab_snapshot_locked(uvm_parent_gpu_t *parent_gpu, uvm_vab_snapshot_t *snapshot)
{
    unsigned mode = vab_mode();

    *snapshot = parent_gpu->vab.sample;
    snapshot->linear_base = uvm_perf_vab_eviction_linear_base;
    snapshot->eviction_enabled = mode == 2 || mode == 3;
    snapshot->lru_enabled = mode == 3;
    snapshot->valid = snapshot->valid && parent_gpu->vab.running &&
                      !READ_ONCE(parent_gpu->vab.stopping) && mode != 0 &&
                      uvm_vab_sample_fresh(NV_GETTIME(), snapshot->sampled_ns,
                                           (NvU64)3 * vab_interval_ms() * NSEC_PER_MSEC);
}

bool uvm_gpu_vab_eviction_requested(uvm_parent_gpu_t *parent_gpu)
{
    return (vab_mode() == 2 || vab_mode() == 3) && uvm_perf_vab_eviction_linear_base != ~0ULL &&
           READ_ONCE(parent_gpu->vab.supported) && READ_ONCE(parent_gpu->vab.running) &&
           !READ_ONCE(parent_gpu->vab.stopping);
}

void uvm_gpu_vab_get_snapshot(uvm_parent_gpu_t *parent_gpu, uvm_vab_snapshot_t *snapshot)
{
    unsigned long flags;

    // No RM calls: this is safe inside PMM and RM's PMA reclaim callbacks.
    spin_lock_irqsave(&parent_gpu->vab.cache_lock, flags);
    vab_snapshot_locked(parent_gpu, snapshot);
    spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
}

static void vab_sample(uvm_parent_gpu_t *parent_gpu)
{
    UvmGpuAccessBitsBufferAlloc raw;
    NvU64 started_ns = NV_GETTIME();
    NvU64 duration_ns;
    NV_STATUS status;
    unsigned long flags;
    unsigned i;
    unsigned enabled = 0;
    unsigned set = 0;
    NvU64 rising = 0;
    NvU64 falling = 0;
    bool history_valid;

    if (!vab_mode()) {
        spin_lock_irqsave(&parent_gpu->vab.cache_lock, flags);
        parent_gpu->vab.sample.valid = false;
        spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
        return;
    }

    // Do not wait on suspend while GPU removal may wait for this worker.
    if (!uvm_down_read_trylock(&g_uvm_global.pm.lock)) {
        spin_lock_irqsave(&parent_gpu->vab.cache_lock, flags);
        ++parent_gpu->vab.pm_skips;
        parent_gpu->vab.sample.valid = false;
        spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
        return;
    }

    mutex_lock(&parent_gpu->vab.dump_lock);
    // A mode-1 worker may have waited behind a diagnostic dump while mode was
    // switched to 0. Never issue a late CURRENT that clears its aggregation.
    if (!vab_mode()) {
        mutex_unlock(&parent_gpu->vab.dump_lock);
        spin_lock_irqsave(&parent_gpu->vab.cache_lock, flags);
        parent_gpu->vab.sample.valid = false;
        spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
        uvm_up_read(&g_uvm_global.pm.lock);
        return;
    }
    status = nvUvmInterfaceAccessBitsDump(parent_gpu->rm_device,
                                          &parent_gpu->vab_info,
                                          UVM_ACCESS_BITS_DUMP_MODE_CURRENT);
    if (status == NV_OK)
        raw = parent_gpu->vab_info;
    mutex_unlock(&parent_gpu->vab.dump_lock);
    duration_ns = NV_GETTIME() - started_ns;

    spin_lock_irqsave(&parent_gpu->vab.cache_lock, flags);
    parent_gpu->vab.last_dump_status = status;
    parent_gpu->vab.dump_duration_ns = duration_ns;
    if (status != NV_OK) {
        ++parent_gpu->vab.dump_errors;
        parent_gpu->vab.sample.valid = false;
        spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
        uvm_up_read(&g_uvm_global.pm.lock);
        return;
    }

    if (raw.bDirtyBits || raw.granularity > 15) {
        parent_gpu->vab.sample.valid = false;
        spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
        uvm_up_read(&g_uvm_global.pm.lock);
        return;
    }
    history_valid = parent_gpu->vab.sample.valid &&
                    parent_gpu->vab.sample.granularity == raw.granularity &&
                    uvm_vab_sample_fresh(started_ns, parent_gpu->vab.sample.sampled_ns,
                                         (NvU64)3 * vab_interval_ms() * NSEC_PER_MSEC);
    // A gap or layout change ends the observation window. Seed currently set
    // bits conservatively rather than carrying old recency across that gap.
    if (!history_valid)
        memset(parent_gpu->vab.last_access_ns, 0, sizeof(parent_gpu->vab.last_access_ns));

    for (i = 0; i < UVM_VAB_WORD_COUNT; ++i) {
        NvU64 mask = raw.enabledMask[i];
        NvU64 bits = raw.currentBits[i] & mask;
        NvU64 common = parent_gpu->vab.samples ? mask & parent_gpu->vab.sample.enabled_mask[i] : 0;
        NvU64 previous = parent_gpu->vab.samples ? parent_gpu->vab.sample.access_bits[i] : 0;
        NvU64 removed = parent_gpu->vab.sample.enabled_mask[i] & ~mask;

        enabled += hweight64(mask);
        set += hweight64(bits);
        if (parent_gpu->vab.samples) {
            rising += hweight64(bits & ~previous & common);
            falling += hweight64(previous & ~bits & common);
        }
        parent_gpu->vab.sample.access_bits[i] = bits;
        parent_gpu->vab.sample.enabled_mask[i] = mask;
        // Track every sampled mode so entering LRU does not lose accesses
        // observed while eviction was disabled. The region experiment refreshes
        // all set bits as recently observed; legacy mode keeps edge-only aging.
        {
            NvU64 touched = uvm_cpu_block_policy_region_migration_enabled() ? bits :
                              (history_valid ? bits & ~(previous & common) : bits);
            while (removed) {
                NvU32 bit = __builtin_ctzll(removed);
                parent_gpu->vab.last_access_ns[i * 64 + bit] = 0;
                removed &= removed - 1;
            }
            while (touched) {
                NvU32 bit = __builtin_ctzll(touched);
                parent_gpu->vab.last_access_ns[i * 64 + bit] = started_ns;
                touched &= touched - 1;
            }
        }
    }
    if (!enabled) {
        parent_gpu->vab.sample.valid = false;
        spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
        uvm_up_read(&g_uvm_global.pm.lock);
        return;
    }
    if (parent_gpu->vab.samples && !rising && !falling)
        ++parent_gpu->vab.unchanged_samples;
    else
        parent_gpu->vab.unchanged_samples = 0;
    ++parent_gpu->vab.samples;
    parent_gpu->vab.rising_bits += rising;
    parent_gpu->vab.falling_bits += falling;
    parent_gpu->vab.enabled_bits = enabled;
    parent_gpu->vab.set_bits = set;
    parent_gpu->vab.sample.granularity = raw.granularity;
    // Use the start time, not dump completion: a root arriving during the
    // dump must not be classified from an observation preceding its arrival.
    parent_gpu->vab.sample.sampled_ns = started_ns;
    parent_gpu->vab.sample.valid = !raw.bDirtyBits && raw.granularity <= 15 && enabled != 0;
    spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
    if (uvm_gpu_vab_eviction_requested(parent_gpu))
        uvm_gpu_vab_kick_readiness(parent_gpu);
    // Publish before releasing PM: suspend invalidates under the write lock,
    // and must not be followed by publication of a pre-suspend observation.
    uvm_up_read(&g_uvm_global.pm.lock);
}

bool uvm_gpu_vab_get_recency(uvm_parent_gpu_t *parent_gpu,
                             const uvm_vab_snapshot_t *snapshot,
                             NvU64 address, NvU64 size, NvU64 ready_ns,
                             NvU64 *last_access_ns)
{
    NvU64 bit_size, end, first, last, bit, newest = ready_ns;
    unsigned long flags;

    if (!last_access_ns || !snapshot || !snapshot->valid || !snapshot->lru_enabled ||
        snapshot->linear_base == ~(NvU64)0 || snapshot->granularity > 15 ||
        !snapshot->sampled_ns || snapshot->sampled_ns < ready_ns || !size ||
        address > ~(NvU64)0 - (size - 1) || address < snapshot->linear_base)
        return false;
    bit_size = ((NvU64)64 * 1024) << snapshot->granularity;
    end = address + size - 1;
    first = (address - snapshot->linear_base) / bit_size;
    last = (end - snapshot->linear_base) / bit_size;
    if (end < snapshot->linear_base || last >= UVM_VAB_WORD_COUNT * 64)
        return false;
    spin_lock_irqsave(&parent_gpu->vab.cache_lock, flags);
    if (!parent_gpu->vab.sample.valid || parent_gpu->vab.sample.sampled_ns != snapshot->sampled_ns ||
        parent_gpu->vab.sample.granularity != snapshot->granularity) {
        spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
        return false;
    }
    for (bit = first; bit <= last; ++bit) {
        NvU64 mask = (NvU64)1 << (bit % 64);
        if (!(snapshot->enabled_mask[bit / 64] & mask)) {
            spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
            return false;
        }
        if (parent_gpu->vab.last_access_ns[bit] > newest)
            newest = parent_gpu->vab.last_access_ns[bit];
    }
    spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
    *last_access_ns = newest;
    return true;
}

static void vab_sample_work(struct work_struct *work)
{
    uvm_parent_gpu_t *parent_gpu = container_of(to_delayed_work(work), uvm_parent_gpu_t, vab.work);

    if (READ_ONCE(parent_gpu->vab.stopping))
        return;
    UVM_ENTRY_VOID(vab_sample(parent_gpu));
    if (!READ_ONCE(parent_gpu->vab.stopping))
        schedule_delayed_work(&parent_gpu->vab.work, msecs_to_jiffies(vab_interval_ms()));
}

void uvm_gpu_vab_init(uvm_parent_gpu_t *parent_gpu)
{
    mutex_init(&parent_gpu->vab.dump_lock);
    spin_lock_init(&parent_gpu->vab.cache_lock);
    INIT_DELAYED_WORK(&parent_gpu->vab.work, vab_sample_work);
    parent_gpu->vab.stopping = true;
    parent_gpu->vab.last_dump_status = NV_ERR_INVALID_STATE;
}

void uvm_gpu_vab_start(uvm_parent_gpu_t *parent_gpu)
{
    unsigned i;
    unsigned enabled = 0;
    unsigned long flags;
    bool supported;

    BUILD_BUG_ON(ARRAY_SIZE(parent_gpu->vab_info.currentBits) != UVM_VAB_WORD_COUNT);
    for (i = 0; i < UVM_VAB_WORD_COUNT; ++i)
        enabled += hweight64(parent_gpu->vab_info.enabledMask[i]);
    supported = parent_gpu->access_bits_supported && !parent_gpu->smc.enabled &&
                parent_gpu->vab_info.accessBitsBufferHandle && !parent_gpu->vab_info.bDirtyBits &&
                parent_gpu->vab_info.granularity <= 15 && enabled != 0;

    spin_lock_irqsave(&parent_gpu->vab.cache_lock, flags);
    parent_gpu->vab.allocated = parent_gpu->vab_info.accessBitsBufferHandle != 0;
    parent_gpu->vab.supported = supported;
    parent_gpu->vab.enabled_bits = enabled;
    memcpy(parent_gpu->vab.sample.enabled_mask, parent_gpu->vab_info.enabledMask,
           sizeof(parent_gpu->vab.sample.enabled_mask));
    parent_gpu->vab.sample.granularity = parent_gpu->vab_info.granularity;
    parent_gpu->vab.running = supported;
    parent_gpu->vab.stopping = !supported;
    spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
    if (!supported) {
        pr_info("VAB observation inactive: unsupported/SMC/dirty configuration or empty enabled mask\n");
        return;
    }
    pr_info("VAB observation: mode=%u granularity=%llu KiB enabled_bits=%u mapping=%s\n",
            vab_mode(), 64ULL << parent_gpu->vab_info.granularity, enabled,
            uvm_perf_vab_eviction_linear_base == ~0ULL ? "unknown" : "explicit linear assumption");
    schedule_delayed_work(&parent_gpu->vab.work, msecs_to_jiffies(vab_interval_ms()));
}

void uvm_gpu_vab_stop(uvm_parent_gpu_t *parent_gpu)
{
    unsigned long flags;

    WRITE_ONCE(parent_gpu->vab.stopping, true);
    cancel_delayed_work_sync(&parent_gpu->vab.work);
    spin_lock_irqsave(&parent_gpu->vab.cache_lock, flags);
    parent_gpu->vab.running = false;
    parent_gpu->vab.sample.valid = false;
    spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
}

void uvm_gpu_vab_invalidate(uvm_parent_gpu_t *parent_gpu)
{
    unsigned long flags;

    spin_lock_irqsave(&parent_gpu->vab.cache_lock, flags);
    parent_gpu->vab.sample.valid = false;
    spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);
}

NV_STATUS uvm_gpu_vab_test_dump(uvm_parent_gpu_t *parent_gpu,
                                UVM_ACCESS_BITS_DUMP_MODE mode,
                                UvmGpuAccessBitsBufferAlloc *snapshot)
{
    NV_STATUS status;

    mutex_lock(&parent_gpu->vab.dump_lock);
    // CURRENT changes aggregation. Keep diagnostic DIFF/AGGREGATE sequences
    // separate from sampling; mode 0 permits the direct test ioctl again.
    if (vab_mode() && READ_ONCE(parent_gpu->vab.running)) {
        status = NV_ERR_IN_USE;
    }
    else {
        status = nvUvmInterfaceAccessBitsDump(parent_gpu->rm_device, &parent_gpu->vab_info, mode);
        if (status == NV_OK)
            *snapshot = parent_gpu->vab_info;
    }
    mutex_unlock(&parent_gpu->vab.dump_lock);
    return status;
}

void uvm_gpu_vab_print(uvm_gpu_t *gpu, struct seq_file *s)
{
    uvm_parent_gpu_t *parent_gpu = gpu->parent;
    uvm_vab_snapshot_t snapshot;
    uvm_pmm_gpu_vab_stats_t roots;
    NvU64 samples, errors, skips, rising, falling, unchanged, duration;
    NvU32 enabled, set;
    NV_STATUS status;
    bool allocated, supported, running;
    unsigned i;
    unsigned long flags;
    NvU64 now = NV_GETTIME();

    spin_lock_irqsave(&parent_gpu->vab.cache_lock, flags);
    vab_snapshot_locked(parent_gpu, &snapshot);
    samples = parent_gpu->vab.samples;
    errors = parent_gpu->vab.dump_errors;
    skips = parent_gpu->vab.pm_skips;
    rising = parent_gpu->vab.rising_bits;
    falling = parent_gpu->vab.falling_bits;
    unchanged = parent_gpu->vab.unchanged_samples;
    duration = parent_gpu->vab.dump_duration_ns;
    enabled = parent_gpu->vab.enabled_bits;
    set = parent_gpu->vab.set_bits;
    status = parent_gpu->vab.last_dump_status;
    allocated = parent_gpu->vab.allocated;
    supported = parent_gpu->vab.supported;
    running = parent_gpu->vab.running && !parent_gpu->vab.stopping;
    spin_unlock_irqrestore(&parent_gpu->vab.cache_lock, flags);

    // Cache and PMM locks are never nested in this order. Take a fixed sample
    // before scanning the child's used roots; readers never call RM.
    uvm_pmm_gpu_vab_stats(&gpu->pmm, &snapshot, &roots);
    seq_printf(s, "mode %u\ninterval_ms %u\nallocated %u\nsupported %u\nworker_running %u\nsampling_active %u\n",
               vab_mode(), vab_interval_ms(), allocated, supported, running, running && vab_mode() != 0);
    seq_printf(s, "mapping_configured %u\nlinear_base 0x%llx\nsample_valid %u\n",
               snapshot.linear_base != ~0ULL, snapshot.linear_base, snapshot.valid);
    seq_printf(s, "selection_active %u\nclassification_scope %s\n",
               (snapshot.eviction_enabled || snapshot.lru_enabled) && snapshot.valid && snapshot.linear_base != ~0ULL,
               snapshot.lru_enabled ? "VAB_recency_full_managed_root" :
               (snapshot.eviction_enabled ? "eligible_full_managed_root" : "physical_region_at_sample"));
    seq_printf(s, "granularity_kib %llu\nbitmap_bits %u\nenabled_bits %u\nset_bits %u\nclear_bits %u\n",
               snapshot.granularity <= 15 ? 64ULL << snapshot.granularity : 0,
               UVM_VAB_WORD_COUNT * 64, enabled, set, enabled - min(enabled, set));
    seq_printf(s, "samples %llu\ndump_errors %llu\npm_skips %llu\nlast_dump_status 0x%08x\n",
               samples, errors, skips, status);
    if (snapshot.sampled_ns && now >= snapshot.sampled_ns)
        seq_printf(s, "sample_age_ms %llu\n", (now - snapshot.sampled_ns) / NSEC_PER_MSEC);
    else
        seq_puts(s, "sample_age_ms unknown\n");
    seq_printf(s, "last_dump_us %llu\nrising_bits %llu\nfalling_bits %llu\nunchanged_samples %llu\n",
               duration / 1000, rising, falling, unchanged);
    seq_printf(s, "used_roots_hot %llu\nused_roots_cold %llu\nused_roots_unknown %llu\n",
               roots.used_hot, roots.used_cold, roots.used_unknown);
    seq_printf(s, "selected_hot %llu\nselected_cold %llu\nselected_unknown %llu\n",
               roots.selected_hot, roots.selected_cold, roots.selected_unknown);
    seq_printf(s, "background_roots_freed %llu\nbackground_failures %llu\n",
               roots.background_roots_freed, roots.background_failures);
    seq_printf(s, "selection_changes %llu\nselection_fallbacks %llu\neligible_roots %llu\nreadiness_pending %llu\n",
               roots.selection_changes, roots.selection_fallbacks, roots.eligible_roots, roots.readiness_pending);
    seq_printf(s, "usage_percent %llu\nprefetch_inflight %llu\nrecovering_watermark %llu\nbackground_active %llu\n",
               roots.usage_percent, roots.prefetch_inflight, roots.recovering_watermark, roots.background_active);
    seq_printf(s, "migration_policy %s\nbackground_policy %s\n",
               uvm_cpu_block_policy_region_migration_enabled() ? "threshold_4MiB_VA_regions" : "legacy",
               uvm_cpu_block_policy_region_migration_enabled() ? "allocation_driven_VAB_LRU" : "VAB_cold_only");
    seq_printf(s, "background_hot_skips %llu\nbackground_unknown_skips %llu\n",
               roots.background_hot_skips, roots.background_unknown_skips);
    seq_printf(s, "prefetch_limit_bytes %llu\nprefetch_demand_reserve_bytes %llu\nprefetch_reserved_bytes %llu\nprefetch_peak_bytes %llu\nprefetch_window_bytes %llu\n",
               roots.prefetch_limit_bytes, roots.prefetch_demand_reserve_bytes,
               roots.prefetch_reserved_bytes, roots.prefetch_peak_bytes, roots.prefetch_window_bytes);
    seq_printf(s, "prefetch_admissions %llu\n", roots.prefetch_admissions);
    seq_printf(s, "prefetch_denials %llu\n", roots.prefetch_denials);
    seq_printf(s, "prefetch_backoffs %llu\n", roots.prefetch_backoffs);
    seq_printf(s, "prefetch_quick_returns %llu\n", roots.prefetch_quick_returns);
    seq_printf(s, "prefetch_capacity_stops %llu\n", roots.prefetch_capacity_stops);
    seq_printf(s, "prefetch_initialized %llu\n", roots.prefetch_initialized);
    seq_printf(s, "prefetch_off %llu\n", roots.prefetch_off);
    seq_printf(s, "prefetch_pressure_pauses %llu\n", roots.prefetch_pressure_pauses);
    seq_printf(s, "prefetch_allocation_failures %llu\n", roots.prefetch_allocation_failures);
    seq_printf(s, "prefetch_prefetched_blocks %llu\n", roots.prefetch_prefetched_blocks);
    seq_printf(s, "prefetch_mature_blocks %llu\n", roots.prefetch_mature_blocks);
    seq_printf(s, "prefetch_mature_returns %llu\n", roots.prefetch_mature_returns);
    seq_printf(s, "prefetch_pending_blocks %llu\n", roots.prefetch_pending_blocks);
    seq_printf(s, "prefetch_pending_returns %llu\n", roots.prefetch_pending_returns);
    seq_printf(s, "prefetch_last_blocks %llu\n", roots.prefetch_last_blocks);
    seq_printf(s, "prefetch_last_returns %llu\n", roots.prefetch_last_returns);
    seq_printf(s, "prefetch_ratio_evaluations %llu\n", roots.prefetch_ratio_evaluations);

    seq_printf(s, "prefetch_discarded_blocks %llu\n", roots.prefetch_discarded_blocks);
    seq_printf(s, "prefetch_discarded_returns %llu\n", roots.prefetch_discarded_returns);
    seq_printf(s, "prefetch_dropped_records %llu\n", roots.prefetch_dropped_records);
    seq_printf(s, "prefetch_dropped_returns %llu\n", roots.prefetch_dropped_returns);
    seq_printf(s, "prefetch_unretired_blocks %llu\n", roots.prefetch_unretired_blocks);
    seq_printf(s, "prefetch_unretired_returns %llu\n", roots.prefetch_unretired_returns);
    seq_printf(s, "prefetch_ready_blocks %llu\n", roots.prefetch_ready_blocks);
    seq_printf(s, "prefetch_ready_returns %llu\n", roots.prefetch_ready_returns);
    seq_puts(s, "prefetch_policy rapid_return_ratio\n");
    seq_printf(s, "prefetch_backoff_enabled %llu\n", roots.prefetch_backoff_enabled);
    seq_printf(s, "prefetch_recovery_enabled %llu\n", roots.prefetch_recovery_enabled);
    seq_printf(s, "prefetch_recoveries %llu\n", roots.prefetch_recoveries);
    seq_printf(s, "prefetch_probes %llu\n", roots.prefetch_probes);
    seq_printf(s, "prefetch_probe_inflight %llu\n", roots.prefetch_probe_inflight);
    seq_printf(s, "prefetch_next_probe_ns %llu\n", roots.prefetch_next_probe_ns);
    seq_printf(s, "prefetch_observation_ns %llu\nprefetch_return_percent %llu\nprefetch_min_blocks %llu\n",
               roots.prefetch_observation_ns, roots.prefetch_return_percent, roots.prefetch_min_blocks);

    // Raw words expose hot/cold bits even without a physical range mapping.
    // Word zero is first; bit i belongs to word i/64 at position i%64.
    seq_puts(s, "enabled_mask");
    for (i = 0; i < UVM_VAB_WORD_COUNT; ++i)
        seq_printf(s, " %016llx", snapshot.enabled_mask[i]);
    seq_puts(s, "\naccess_bits");
    for (i = 0; i < UVM_VAB_WORD_COUNT; ++i)
        seq_printf(s, " %016llx", snapshot.access_bits[i]);
    seq_puts(s, "\n");
}
