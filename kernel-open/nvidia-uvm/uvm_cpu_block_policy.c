/*******************************************************************************
    Copyright (c) 2026 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#include "uvm_cpu_block_policy.h"
#include "uvm_global.h"
#include "uvm_gpu.h"
#include "uvm_linux.h"
#include "uvm_perf_thrashing.h"
#include "uvm_va_block.h"
#include "uvm_va_range.h"
#include "uvm_va_space.h"

static atomic64_t cpu_prefetch_next_range_id = ATOMIC64_INIT(0);

static unsigned uvm_cpu_preferred_blocks_enable __read_mostly;
static unsigned uvm_cpu_preferred_block_promotion_enable __read_mostly = 0;
static unsigned uvm_cpu_preferred_prefetch_chunk_mb __read_mostly = 64;
static atomic64_t uvm_cpu_preferred_block_promotions = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_range_prefetches = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_prefetch_retries = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_prefetched_bytes = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_trigger_prefetched_bytes = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_trigger_indicated_bytes = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_trigger_extra_bytes = ATOMIC64_INIT(0);
static bool uvm_perf_access_counter_indicated_pages_enable;
static atomic64_t uvm_cpu_preferred_neighbor_prefetched_bytes = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_prefetch_capacity_stops = ATOMIC64_INIT(0);
static bool uvm_perf_access_counter_admission_enable = true;
static unsigned uvm_perf_access_counter_admission_pressure_percent = 95;
static unsigned uvm_perf_access_counter_admission_observations = 2;
static bool uvm_perf_access_counter_admission_coverage_enable;
static unsigned uvm_perf_access_counter_admission_coverage_percent = 25;
static unsigned uvm_perf_access_counter_admission_sparse_observations = 4;
static atomic64_t uvm_cpu_preferred_admission_coverage_broad = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_admission_coverage_sparse = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_admission_coverage_deferred = ATOMIC64_INIT(0);
static bool uvm_perf_access_counter_admission_eviction_enable = false;
static bool uvm_perf_access_counter_admission_distance_stats_enable = true;
static unsigned uvm_perf_access_counter_admission_eviction_mb = 256;
static unsigned uvm_perf_access_counter_admission_eviction_extra = 2;
static atomic64_t uvm_cpu_preferred_admission_eviction_probations = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_admission_eviction_deferred = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_admission_return_bytes = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_return_probations[5] = { ATOMIC64_INIT(0) };
static atomic64_t uvm_cpu_preferred_return_grants[5] = { ATOMIC64_INIT(0) };
static atomic64_t uvm_cpu_preferred_return_populations[5] = { ATOMIC64_INIT(0) };
static atomic64_t uvm_cpu_preferred_return_population_bytes[5] = { ATOMIC64_INIT(0) };
static atomic64_t uvm_cpu_preferred_admission_requests = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_admission_deferred = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_admission_low_pressure = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_admission_repeated = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_admission_state_failures = ATOMIC64_INIT(0);

static int promotion_count_get(char *buffer, const struct kernel_param *kp)
{
    (void)kp;

    return scnprintf(buffer,
                     PAGE_SIZE,
                     "%lld\n",
                     (long long)atomic64_read(&uvm_cpu_preferred_block_promotions));
}

static const struct kernel_param_ops promotion_count_ops = {
    .get = promotion_count_get,
};

static int atomic64_count_get(char *buffer, const struct kernel_param *kp)
{
    atomic64_t *counter = kp->arg;

    return scnprintf(buffer, PAGE_SIZE, "%lld\n", (long long)atomic64_read(counter));
}

static const struct kernel_param_ops atomic64_count_ops = {
    .get = atomic64_count_get,
};

module_param(uvm_perf_access_counter_admission_enable, bool, S_IRUGO);
module_param(uvm_perf_access_counter_admission_coverage_enable, bool, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_admission_coverage_enable,
                 "Require more fresh observations for narrow block coverage under pressure (default false).");
module_param(uvm_perf_access_counter_admission_coverage_percent, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_admission_coverage_percent,
                 "Fresh notification coverage percent qualifying for normal block admission (default 25, range 1-100).");
module_param(uvm_perf_access_counter_admission_sparse_observations, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_admission_sparse_observations,
                 "Minimum fresh observations for narrow block coverage (default 4, range 2-8).");
module_param_cb(uvm_cpu_preferred_admission_coverage_broad, &atomic64_count_ops, &uvm_cpu_preferred_admission_coverage_broad, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_admission_coverage_broad,
                 "Pressure admission grants with broad notification coverage, not completed transfers.");
module_param_cb(uvm_cpu_preferred_admission_coverage_sparse, &atomic64_count_ops, &uvm_cpu_preferred_admission_coverage_sparse, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_admission_coverage_sparse,
                 "Pressure admission grants with narrow notification coverage, not completed transfers.");
module_param_cb(uvm_cpu_preferred_admission_coverage_deferred, &atomic64_count_ops, &uvm_cpu_preferred_admission_coverage_deferred, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_admission_coverage_deferred,
                 "Denied pressure decisions with narrow coverage, including duplicate reports and base probation.");
module_param(uvm_perf_access_counter_indicated_pages_enable, bool, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_indicated_pages_enable,
                 "Under pressure, limit optional trigger population to counter-indicated pages (default false).");
module_param_cb(uvm_cpu_preferred_trigger_indicated_bytes, &atomic64_count_ops, &uvm_cpu_preferred_trigger_indicated_bytes, S_IRUGO);
module_param_cb(uvm_cpu_preferred_trigger_extra_bytes, &atomic64_count_ops, &uvm_cpu_preferred_trigger_extra_bytes, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_admission_enable,
                 "Pressure-aware admission before CPU-preferred counter promotion (default true). Disables legacy refault path.");
module_param(uvm_perf_access_counter_admission_pressure_percent, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_admission_pressure_percent,
                 "PMA usage percentage requiring reaccess after a completed counter clear (default 95, clamped 1..100).");
module_param(uvm_perf_access_counter_admission_observations, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_admission_observations,
                 "Base clear-separated observations under pressure (default 2, clamped 2..8).");
module_param(uvm_perf_access_counter_admission_eviction_enable, bool, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_admission_eviction_enable,
                 "Extra pressure admission probation for recently completely evicted logical blocks (default false).");
module_param(uvm_perf_access_counter_admission_distance_stats_enable, bool, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_admission_distance_stats_enable,
                 "Measure eviction-to-return population distance independently of extra probation (default true).");
module_param(uvm_perf_access_counter_admission_eviction_mb, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_admission_eviction_mb,
                 "GPU CPU-policy population distance in MiB defining a recent eviction (default 256; zero disables extra probation).");
module_param(uvm_perf_access_counter_admission_eviction_extra, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_access_counter_admission_eviction_extra,
                 "Additional clear-separated observations for recent returns (default 2, total capped at 8).");
module_param_cb(uvm_cpu_preferred_admission_eviction_probations, &atomic64_count_ops, &uvm_cpu_preferred_admission_eviction_probations, S_IRUGO);
module_param_cb(uvm_cpu_preferred_admission_eviction_deferred, &atomic64_count_ops, &uvm_cpu_preferred_admission_eviction_deferred, S_IRUGO);
module_param_cb(uvm_cpu_preferred_admission_return_bytes, &atomic64_count_ops, &uvm_cpu_preferred_admission_return_bytes, S_IRUGO);

#define UVM_RETURN_DISTANCE_COUNTERS(label, index) \
    module_param_cb(uvm_cpu_preferred_return_##label##_probations, &atomic64_count_ops, \
                    &uvm_cpu_preferred_return_probations[index], S_IRUGO); \
    module_param_cb(uvm_cpu_preferred_return_##label##_grants, &atomic64_count_ops, \
                    &uvm_cpu_preferred_return_grants[index], S_IRUGO); \
    module_param_cb(uvm_cpu_preferred_return_##label##_populations, &atomic64_count_ops, \
                    &uvm_cpu_preferred_return_populations[index], S_IRUGO); \
    module_param_cb(uvm_cpu_preferred_return_##label##_population_bytes, &atomic64_count_ops, \
                    &uvm_cpu_preferred_return_population_bytes[index], S_IRUGO)

UVM_RETURN_DISTANCE_COUNTERS(lt64m, 0);
UVM_RETURN_DISTANCE_COUNTERS(64m_256m, 1);
UVM_RETURN_DISTANCE_COUNTERS(256m_1g, 2);
UVM_RETURN_DISTANCE_COUNTERS(1g_4g, 3);
UVM_RETURN_DISTANCE_COUNTERS(ge4g, 4);
#undef UVM_RETURN_DISTANCE_COUNTERS
module_param_cb(uvm_cpu_preferred_admission_requests, &atomic64_count_ops, &uvm_cpu_preferred_admission_requests, S_IRUGO);
module_param_cb(uvm_cpu_preferred_admission_deferred, &atomic64_count_ops, &uvm_cpu_preferred_admission_deferred, S_IRUGO);
module_param_cb(uvm_cpu_preferred_admission_low_pressure, &atomic64_count_ops, &uvm_cpu_preferred_admission_low_pressure, S_IRUGO);
module_param_cb(uvm_cpu_preferred_admission_repeated, &atomic64_count_ops, &uvm_cpu_preferred_admission_repeated, S_IRUGO);
module_param_cb(uvm_cpu_preferred_admission_state_failures, &atomic64_count_ops, &uvm_cpu_preferred_admission_state_failures, S_IRUGO);

bool uvm_cpu_block_policy_counter_admission_enabled(void)
{
    return uvm_perf_access_counter_admission_enable;
}

bool uvm_cpu_block_policy_counter_pressure(uvm_gpu_t *gpu)
{
    return uvm_cpu_block_policy_counter_admission_enabled() &&
           uvm_pmm_gpu_counter_admission_pressure(&gpu->pmm,
               clamp(uvm_perf_access_counter_admission_pressure_percent, 1U, 100U));
}

static void counter_admission_record_return_grant(uvm_va_block_gpu_state_t *state, uvm_gpu_t *gpu)
{
    if (state && state->counter_admission_history.evicted &&
        uvm_perf_access_counter_admission_distance_stats_enable) {
        NvU64 population = (NvU64)atomic64_read(&gpu->counter_admission_population_bytes);
        unsigned bucket = uvm_counter_admission_distance_bucket(
                              population - state->counter_admission_history.population_at_eviction);
        atomic64_inc(&uvm_cpu_preferred_return_grants[bucket]);
    }
}

bool uvm_cpu_block_policy_admit_counter(uvm_va_block_t *va_block, uvm_gpu_t *gpu,
                                       NvU32 buffer, NvU64 submitted, NvU64 fresh,
                                       const uvm_page_mask_t *accessed_pages)
{
    uvm_va_block_gpu_state_t *state;
    bool pressure;
    bool admitted;
    NvU32 required;
    bool narrow = false;

    uvm_assert_mutex_locked(&va_block->lock);
    if (!uvm_cpu_block_policy_counter_admission_enabled() || uvm_va_block_is_hmm(va_block) ||
        !va_block->managed_range ||
        !uvm_cpu_block_policy_should_service_4k(va_block->managed_range, gpu))
        return true;

    atomic64_inc(&uvm_cpu_preferred_admission_requests);
    pressure = uvm_cpu_block_policy_counter_pressure(gpu);
    state = uvm_va_block_gpu_state_get(va_block, gpu->id);
    if (state && !uvm_page_mask_empty(&state->resident))
        state->counter_admission_history.evicted = false;
    if (!pressure) {
        if (state) {
            state->counter_admission.pending = false;
            state->counter_admission.observations = 0;
        }
        atomic64_inc(&uvm_cpu_preferred_admission_low_pressure);
        counter_admission_record_return_grant(state, gpu);
        return true;
    }
    if (!state)
        state = uvm_va_block_gpu_state_get_alloc(va_block, gpu);
    if (!state) {
        atomic64_inc(&uvm_cpu_preferred_admission_state_failures);
        atomic64_inc(&uvm_cpu_preferred_admission_deferred);
        return false;
    }
    required = clamp(uvm_perf_access_counter_admission_observations, 2U, 8U);
    if ((uvm_perf_access_counter_admission_eviction_enable ||
         uvm_perf_access_counter_admission_distance_stats_enable) &&
        (!state->counter_admission.pending || state->counter_admission.buffer != buffer)) {
        NvU64 population = (NvU64)atomic64_read(&gpu->counter_admission_population_bytes);
        required = uvm_counter_admission_required(&state->counter_admission_history,
                      population,
                      (NvU64)uvm_perf_access_counter_admission_eviction_mb * 1024 * 1024,
                      required, uvm_perf_access_counter_admission_eviction_enable ?
                                    uvm_perf_access_counter_admission_eviction_extra : 0);
        if (uvm_perf_access_counter_admission_distance_stats_enable && state->counter_admission_history.evicted) {
            unsigned bucket = uvm_counter_admission_distance_bucket(
                                  population - state->counter_admission_history.population_at_eviction);
            atomic64_inc(&uvm_cpu_preferred_return_probations[bucket]);
        }
        if (required > clamp(uvm_perf_access_counter_admission_observations, 2U, 8U))
            atomic64_inc(&uvm_cpu_preferred_admission_eviction_probations);
    }
    if (uvm_perf_access_counter_admission_coverage_enable) {
        bool new_probation = !state->counter_admission.pending || state->counter_admission.buffer != buffer;
        if (new_probation) {
            uvm_page_mask_zero(&state->counter_admission_pages);
            state->counter_admission.coverage_base_required = required;
        }
        // Only evidence separated by a completed clear may widen coverage.
        // Duplicate queued reports neither advance probation nor add pages.
        if (accessed_pages && uvm_counter_admission_fresh(&state->counter_admission, buffer, fresh))
            uvm_page_mask_or(&state->counter_admission_pages, &state->counter_admission_pages, accessed_pages);
        narrow = (NvU64)uvm_page_mask_weight(&state->counter_admission_pages) * 100 <
                 (NvU64)uvm_va_block_num_cpu_pages(va_block) *
                     clamp(uvm_perf_access_counter_admission_coverage_percent, 1U, 100U);
        required = state->counter_admission.coverage_base_required;
        if (narrow)
            required = max(required, clamp(uvm_perf_access_counter_admission_sparse_observations, 2U, 8U));
        // Broadening the evidence may restore the base target mid-probation.
        // The original eviction-based target stays frozen separately.
        if (!new_probation)
            state->counter_admission.required = required;
    }
    admitted = uvm_counter_admission_decide(&state->counter_admission, true, buffer, submitted, fresh, required);
    if (uvm_perf_access_counter_admission_coverage_enable) {
        if (admitted) {
            if (narrow)
                atomic64_inc(&uvm_cpu_preferred_admission_coverage_sparse);
            else
                atomic64_inc(&uvm_cpu_preferred_admission_coverage_broad);
        }
        else if (narrow)
            atomic64_inc(&uvm_cpu_preferred_admission_coverage_deferred);
    }
    if (!admitted && (uvm_perf_access_counter_admission_coverage_enable ?
                         state->counter_admission.coverage_base_required : state->counter_admission.required) >
                         clamp(uvm_perf_access_counter_admission_observations, 2U, 8U))
        atomic64_inc(&uvm_cpu_preferred_admission_eviction_deferred);
    if (admitted) {
        atomic64_inc(&uvm_cpu_preferred_admission_repeated);
        counter_admission_record_return_grant(state, gpu);
    }
    else
        atomic64_inc(&uvm_cpu_preferred_admission_deferred);
    return admitted;
}

/**
 * Introduced module params + description
 * Define new module params here
 */
module_param(uvm_cpu_preferred_blocks_enable, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_blocks_enable, "Keep managed allocations CPU-preferred and enable access-counter-guided migration policy.");
module_param(uvm_cpu_preferred_block_promotion_enable, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_cpu_preferred_block_promotion_enable, "Promote complete populated 2MB VA blocks on GPU access-counter notifications. " "Zero leaves speculative prefetch enabled without block promotion.");
module_param(uvm_cpu_preferred_prefetch_chunk_mb, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_cpu_preferred_prefetch_chunk_mb, "Access-counter-guided prefetch chunk size in MiB. Non-zero values are rounded up to 2MB; " "zero disables speculative chunk prefetch.");
module_param_cb(uvm_cpu_preferred_block_promotions, &promotion_count_ops, NULL, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_block_promotions, "Number of successful access-counter VA-block promotion service operations.");
module_param_cb(uvm_cpu_preferred_range_prefetches, &atomic64_count_ops, &uvm_cpu_preferred_range_prefetches,S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_range_prefetches,"Number of access-counter-guided managed allocation chunk prefetches started.");
module_param_cb(uvm_cpu_preferred_prefetch_retries, &atomic64_count_ops, &uvm_cpu_preferred_prefetch_retries, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_prefetch_retries, "Number of capacity-stopped managed allocation prefetches retried.");
module_param_cb(uvm_cpu_preferred_prefetched_bytes, &atomic64_count_ops, &uvm_cpu_preferred_prefetched_bytes, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_prefetched_bytes, "Number of bytes moved by access-counter-guided managed allocation chunk prefetches.");
module_param_cb(uvm_cpu_preferred_trigger_prefetched_bytes, &atomic64_count_ops, &uvm_cpu_preferred_trigger_prefetched_bytes, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_trigger_prefetched_bytes, "Prefetch bytes attributed to the triggering VA block, including its unaccessed pages; subset of prefetched_bytes.");
module_param_cb(uvm_cpu_preferred_neighbor_prefetched_bytes, &atomic64_count_ops, &uvm_cpu_preferred_neighbor_prefetched_bytes, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_neighbor_prefetched_bytes, "Prefetch bytes attributed to additional VA blocks; subset of prefetched_bytes.");
module_param_cb(uvm_cpu_preferred_prefetch_capacity_stops, &atomic64_count_ops, &uvm_cpu_preferred_prefetch_capacity_stops, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_prefetch_capacity_stops, "Number of managed allocation prefetches stopped when free VRAM was exhausted.");


// check if block policy is enabled
bool uvm_cpu_block_policy_enabled(void)
{
    return uvm_cpu_preferred_blocks_enable != 0;
}

// check if gpu is supported
static bool gpu_supported(uvm_va_space_t *va_space, uvm_gpu_t *gpu)
{
    if (!gpu || !gpu->parent->access_counters_supported || gpu->parent->is_integrated_gpu)
        return false;

    return uvm_processor_mask_test(&va_space->accessible_from[uvm_id_value(UVM_ID_CPU)], gpu->id);
}


void uvm_cpu_block_policy_init_range(uvm_va_range_managed_t *managed_range)
{
    uvm_va_space_t *va_space = managed_range->va_range.va_space;
    uvm_gpu_t *gpu;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (!uvm_cpu_block_policy_enabled() || g_uvm_global.conf_computing_enabled)
        return;

    managed_range->cpu_prefetch_range_id = (NvU64)atomic64_inc_return(&cpu_prefetch_next_range_id);
    managed_range->cpu_access_counter_policy = true;
    managed_range->policy.preferred_location = UVM_ID_CPU;
    managed_range->policy.preferred_nid = NUMA_NO_NODE;

    for_each_va_space_gpu(gpu, va_space) {
        if (gpu_supported(va_space, gpu))
            uvm_processor_mask_set(&managed_range->policy.accessed_by, gpu->id);
    }
}

bool uvm_cpu_block_policy_should_add_accessed_by(uvm_va_range_managed_t *managed_range,
                                                uvm_gpu_t *gpu)
{
    uvm_va_space_t *va_space = managed_range->va_range.va_space;
    uvm_va_block_t *va_block;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (!managed_range->cpu_access_counter_policy ||
        !UVM_ID_IS_CPU(managed_range->policy.preferred_location) ||
        !gpu_supported(va_space, gpu)) {
        return false;
    }

    // A GPU may be unregistered and registered again in the same VA space.
    // Permit the newly registered instance to trigger each block again.
    for_each_va_block_in_va_range(managed_range, va_block) {
        uvm_processor_mask_clear_atomic(&va_block->gpu_prefetch_started, gpu->id);
        uvm_processor_mask_clear_atomic(&va_block->gpu_prefetch_retry, gpu->id);
    }

    uvm_processor_mask_set(&managed_range->policy.accessed_by, gpu->id);
    return true;
}

bool uvm_cpu_block_policy_should_service_4k(uvm_va_range_managed_t *managed_range,
                                           uvm_gpu_t *gpu)
{
    uvm_va_space_t *va_space = managed_range->va_range.va_space;

    uvm_assert_rwsem_locked(&va_space->lock);

    return managed_range->cpu_access_counter_policy &&
           UVM_ID_IS_CPU(managed_range->policy.preferred_location) &&
           gpu_supported(va_space, gpu) &&
           uvm_processor_mask_test(&managed_range->policy.accessed_by, gpu->id);
}

bool uvm_cpu_block_policy_should_promote(uvm_va_block_t *va_block, uvm_gpu_t *gpu)
{
    uvm_va_range_managed_t *managed_range;

    if (!READ_ONCE(uvm_cpu_preferred_block_promotion_enable) ||
        !gpu ||
        uvm_va_block_is_hmm(va_block)) {
        return false;
    }

    uvm_assert_mutex_locked(&va_block->lock);
    managed_range = va_block->managed_range;
    if (!managed_range || !managed_range->cpu_access_counter_policy)
        return false;

    return uvm_cpu_block_policy_should_service_4k(managed_range, gpu);
}

static bool should_handle_gpu_signal(uvm_va_block_t *va_block, uvm_gpu_t *gpu)
{
    uvm_va_range_managed_t *managed_range;

    if (!gpu || uvm_va_block_is_hmm(va_block))
        return false;

    managed_range = va_block->managed_range;

    return managed_range &&
           managed_range->cpu_access_counter_policy &&
           UVM_ID_IS_CPU(managed_range->policy.preferred_location) &&
           gpu_supported(managed_range->va_range.va_space, gpu);
}

bool uvm_cpu_block_policy_should_promote_on_fault(uvm_va_block_t *va_block, uvm_gpu_t *gpu)
{
    return READ_ONCE(uvm_cpu_preferred_block_promotion_enable) &&
           should_handle_gpu_signal(va_block, gpu);
}

static NV_STATUS prefetch_block(uvm_va_block_t *va_block,
                                uvm_gpu_t *gpu,
                                struct mm_struct *mm,
                                uvm_va_block_context_t *block_context,
                                uvm_tracker_t *prefetch_tracker,
                                NvU64 *prefetched_bytes,
                                uvm_prefetch_reservation_t *reservation,
                                bool *capacity_stop,
                                bool speculative_neighbor,
                                const uvm_page_mask_t *indicated_pages,
                                bool limit_to_indicated)
{
    NV_STATUS status;
    uvm_va_block_retry_t retry;
    uvm_page_mask_t prefetch_mask;
    const uvm_page_mask_t *cpu_resident;
    const uvm_page_mask_t *gpu_resident;
    const uvm_page_mask_t *thrashing_pages;

    *capacity_stop = false;
    uvm_mutex_lock(&va_block->lock);

    cpu_resident = uvm_va_block_resident_mask_get(va_block, UVM_ID_CPU, NUMA_NO_NODE);
    if (!cpu_resident) {
        status = NV_OK;
        goto out_unlock;
    }

    uvm_page_mask_copy(&prefetch_mask, cpu_resident);

    gpu_resident = uvm_va_block_resident_mask_get(va_block, gpu->id, NUMA_NO_NODE);
    if (gpu_resident) {
        if (!uvm_page_mask_empty(gpu_resident)) {
            uvm_va_block_gpu_state_t *state = uvm_va_block_gpu_state_get(va_block, gpu->id);
            UVM_ASSERT(state);
            state->counter_admission_history.evicted = false;
        }
        uvm_page_mask_andnot(&prefetch_mask, &prefetch_mask, gpu_resident);
    }

    // Do not speculatively move pages which UVM's thrashing policy is
    // currently protecting from migration.
    thrashing_pages = uvm_perf_thrashing_get_thrashing_pages(va_block);
    if (thrashing_pages)
        uvm_page_mask_andnot(&prefetch_mask, &prefetch_mask, thrashing_pages);

    if (limit_to_indicated) {
        if (!indicated_pages)
            uvm_page_mask_zero(&prefetch_mask);
        else
            uvm_page_mask_and(&prefetch_mask, &prefetch_mask, indicated_pages);
    }

    if (uvm_page_mask_empty(&prefetch_mask)) {
        status = NV_OK;
        goto out_unlock;
    }

    uvm_va_block_context_init(block_context, mm);
    uvm_page_mask_zero(&block_context->make_resident.pages_changed_residency);
    uvm_processor_mask_zero(&block_context->make_resident.all_involved_processors);

    uvm_va_block_retry_init(&retry);
    retry.allow_eviction = false;

    status = uvm_va_block_make_resident(va_block,
                                        &retry,
                                        block_context,
                                        gpu->id,
                                        uvm_va_block_region_from_block(va_block),
                                        &prefetch_mask,
                                        NULL,
                                        UVM_MAKE_RESIDENT_CAUSE_PREFETCH);

    if (status == NV_OK) {
        NvU64 moved = (NvU64)uvm_page_mask_weight(&block_context->make_resident.pages_changed_residency) * PAGE_SIZE;
        *prefetched_bytes += moved;
        if (!speculative_neighbor && indicated_pages && moved) {
            uvm_page_mask_t overlap;
            NvU64 indicated;
            uvm_page_mask_and(&overlap, &block_context->make_resident.pages_changed_residency, indicated_pages);
            indicated = (NvU64)uvm_page_mask_weight(&overlap) * PAGE_SIZE;
            atomic64_add(indicated, &uvm_cpu_preferred_trigger_indicated_bytes);
            atomic64_add(moved - indicated, &uvm_cpu_preferred_trigger_extra_bytes);
        }
        if (moved && (uvm_perf_access_counter_admission_eviction_enable ||
                      uvm_perf_access_counter_admission_distance_stats_enable)) {
            uvm_va_block_gpu_state_t *state = uvm_va_block_gpu_state_get(va_block, gpu->id);
            NvU64 population = (NvU64)atomic64_add_return(moved, &gpu->counter_admission_population_bytes) - moved;
            UVM_ASSERT(state);
            if (state->counter_admission_history.evicted) {
                atomic64_add(moved, &uvm_cpu_preferred_admission_return_bytes);
                if (uvm_perf_access_counter_admission_distance_stats_enable) {
                    unsigned bucket = uvm_counter_admission_distance_bucket(
                                          population - state->counter_admission_history.population_at_eviction);
                    atomic64_inc(&uvm_cpu_preferred_return_populations[bucket]);
                    atomic64_add(moved, &uvm_cpu_preferred_return_population_bytes[bucket]);
                }
            }
            state->counter_admission_history.evicted = false;
        }
        if (moved && speculative_neighbor) {
            uvm_va_block_gpu_state_t *gpu_state = uvm_va_block_gpu_state_get(va_block, gpu->id);
            UVM_ASSERT(gpu_state);
            uvm_pmm_gpu_refault_population(&gpu->pmm, va_block->managed_range->cpu_prefetch_range_id,
                                           va_block->start);
            // A submission timestamp is sufficient for coarse churn detection;
            // it is never treated as hardware access/reuse evidence.
            gpu_state->speculative_prefetch_ns = NV_GETTIME();
            gpu_state->speculative_evicted_ns = 0;
            gpu_state->speculative_cohort = uvm_pmm_gpu_prefetch_record(&gpu->pmm,
                                                                       gpu_state->speculative_prefetch_ns,
                                                                       reservation);
        }

        // make_resident updates physical residency but does not install a
        // destination GPU PTE. Map the prefetched pages locally now so their
        // first use does not fall back to an AccessedBy system-memory mapping.
        do {
            status = uvm_va_block_map(va_block,
                                      block_context,
                                      gpu->id,
                                      uvm_va_block_region_from_block(va_block),
                                      &prefetch_mask,
                                      UVM_PROT_READ_WRITE_ATOMIC,
                                      UvmEventMapRemoteCauseInvalid,
                                      &va_block->tracker);
        } while (status == NV_ERR_MORE_PROCESSING_REQUIRED);
    }
    else {
        *capacity_stop = retry.no_eviction_allocation_failed;
    }

    // Collect dependencies while the block tracker is locked, including any
    // work submitted before an error. The chunk waits once after submission.
    {
        NV_STATUS tracker_status = uvm_tracker_add_tracker_safe(prefetch_tracker, &va_block->tracker);
        if (status == NV_OK)
            status = tracker_status;
    }

    uvm_mutex_unlock(&va_block->lock);
    uvm_va_block_retry_deinit(&retry, va_block);
    return status;

out_unlock:
    uvm_mutex_unlock(&va_block->lock);
    return status;
}

void uvm_cpu_block_policy_prefetch_on_signal(uvm_va_block_t *trigger_block,
                                             uvm_gpu_t *gpu,
                                             struct mm_struct *mm,
                                             NvU16 region_mask,
                                             const uvm_page_mask_t *indicated_pages)
{
    uvm_va_range_managed_t *managed_range = trigger_block->managed_range;
    uvm_va_space_t *va_space;
    uvm_va_block_context_t *block_context = NULL;
    size_t block_count;
    size_t chunk_block_count;
    size_t end_index;
    size_t start_index;
    size_t trigger_index;
    size_t block_index;
    size_t candidate_index;
    size_t sparse_indices[32];
    size_t sparse_count = 0;
    NvU64 chunk_bytes;
    NvU64 prefetched_bytes = 0;
    NvU64 trigger_prefetched_bytes = 0;
    NvU64 reserved_bytes = 0;
    uvm_prefetch_reservation_t reservation = {0};
    unsigned prefetch_chunk_mb = READ_ONCE(uvm_cpu_preferred_prefetch_chunk_mb);
    bool retry = false;
    bool capacity_stopped = false;
    bool refault_mode = uvm_pmm_gpu_refault_enabled();
    bool prefetch_admitted = false;
    bool rearm_after_completion = false;
    bool limit_to_indicated = false;
    bool prefetch_success = true;
    uvm_tracker_t prefetch_tracker = UVM_TRACKER_INIT();

    if (!managed_range ||
        prefetch_chunk_mb == 0 ||
        !uvm_pmm_gpu_speculative_prefetch_enabled() ||
        !should_handle_gpu_signal(trigger_block, gpu)) {
        return;
    }

    va_space = managed_range->va_range.va_space;
    uvm_assert_rwsem_locked_read(&va_space->lock);

    // Demand service has already run before this callback. A rapid return of a
    // speculatively populated block is feedback to reduce future speculation,
    // not a reason to delay that demand or to infer cache-hit behavior.
    uvm_mutex_lock(&trigger_block->lock);
    {
        uvm_va_block_gpu_state_t *gpu_state = uvm_va_block_gpu_state_get(trigger_block, gpu->id);
        NvU64 now = NV_GETTIME();
        if (gpu_state && gpu_state->speculative_evicted_ns) {
            if (now >= gpu_state->speculative_evicted_ns &&
                now - gpu_state->speculative_evicted_ns <= uvm_pmm_gpu_prefetch_feedback_window_ns())
                uvm_pmm_gpu_prefetch_return(&gpu->pmm, gpu_state->speculative_cohort, now);
            gpu_state->speculative_evicted_ns = 0;
            gpu_state->speculative_cohort = 0;
        }
    }
    uvm_mutex_unlock(&trigger_block->lock);

    if (uvm_processor_mask_test(&trigger_block->gpu_prefetch_started, gpu->id)) {
        if (!uvm_processor_mask_test(&trigger_block->gpu_prefetch_retry, gpu->id) ||
            uvm_pmm_gpu_eager_eviction_watermark_reached(&gpu->pmm)) {
            return;
        }

        // Claim this retry before clearing started. This prevents concurrent
        // GPU service threads from both turning the retry into a new attempt.
        if (!uvm_processor_mask_test_and_clear_atomic(&trigger_block->gpu_prefetch_retry, gpu->id))
            return;

        uvm_processor_mask_clear_atomic(&trigger_block->gpu_prefetch_started, gpu->id);
        retry = true;
    }

    if (uvm_processor_mask_test_and_set_atomic(&trigger_block->gpu_prefetch_started, gpu->id))
        goto out;

    block_count = uvm_va_range_num_blocks(managed_range);
    trigger_index = uvm_va_range_block_index(managed_range, trigger_block->start);
    chunk_bytes = (NvU64)prefetch_chunk_mb * UVM_SIZE_1MB;
    chunk_block_count = (chunk_bytes + UVM_VA_BLOCK_SIZE - 1) / UVM_VA_BLOCK_SIZE;
    chunk_block_count = min(chunk_block_count, block_count);

    // An admitted high-pressure report may populate its trigger, but must not
    // pull in neighbors which have not supplied their own admission evidence.
    if (uvm_cpu_block_policy_counter_pressure(gpu)) {
        chunk_block_count = min_t(size_t, chunk_block_count, 1);
        region_mask = 0;
        limit_to_indicated = uvm_perf_access_counter_indicated_pages_enable && !refault_mode;
    }

    if (refault_mode) {
        unsigned neighbors = uvm_pmm_gpu_refault_neighbors(&gpu->pmm,
                                                          managed_range->cpu_prefetch_range_id,
                                                          trigger_block->start);
        NvU64 base = UVM_ALIGN_DOWN(trigger_block->start, UVM_VA_BLOCK_SIZE) -
                         (NvU64)trigger_index * UVM_VA_BLOCK_SIZE;
        NvU64 tile = trigger_block->start >> 26;
        size_t distance;
        // Only additional neighbors are speculative. Native demand has already
        // serviced the trigger's accessed pages; do not fill the rest of it.
        for (distance = 1; distance < 32 && sparse_count < neighbors; ++distance) {
            size_t next = trigger_index + distance;
            if (next < block_count && ((base + (NvU64)next * UVM_VA_BLOCK_SIZE) >> 26) == tile)
                sparse_indices[sparse_count++] = next;
            if (sparse_count < neighbors && trigger_index >= distance) {
                next = trigger_index - distance;
                if (((base + (NvU64)next * UVM_VA_BLOCK_SIZE) >> 26) == tile)
                    sparse_indices[sparse_count++] = next;
            }
        }
        chunk_block_count = min(chunk_block_count, sparse_count);
        if (!chunk_block_count)
            goto out;
    }
    else if (region_mask) {
        NvU64 range_block_base = UVM_ALIGN_DOWN(trigger_block->start, UVM_VA_BLOCK_SIZE) -
                                   (NvU64)trigger_index * UVM_VA_BLOCK_SIZE;
        NvU64 tile_base = UVM_ALIGN_DOWN(trigger_block->start, 64ULL * UVM_SIZE_1MB);
        NvU64 tile_last = tile_base + 64ULL * UVM_SIZE_1MB - 1;
        size_t first = tile_base > range_block_base ?
                           (size_t)((tile_base - range_block_base) / UVM_VA_BLOCK_SIZE) : 0;
        size_t outer = min(block_count,
                            (size_t)((tile_last - range_block_base) / UVM_VA_BLOCK_SIZE + 1));
        size_t sibling = trigger_index;
        NvU64 trigger_base = range_block_base + (NvU64)trigger_index * UVM_VA_BLOCK_SIZE;

        sparse_indices[sparse_count++] = trigger_index;
        // Keep the other 2MiB half of the triggering 4MiB region first.
        if (trigger_base & UVM_VA_BLOCK_SIZE) {
            if (trigger_index > first)
                sibling = trigger_index - 1;
        }
        else if (trigger_index + 1 < outer) {
            sibling = trigger_index + 1;
        }
        if (sibling != trigger_index)
            sparse_indices[sparse_count++] = sibling;
        for (candidate_index = first; candidate_index < outer; ++candidate_index) {
            NvU64 address = range_block_base + (NvU64)candidate_index * UVM_VA_BLOCK_SIZE;
            unsigned region = (unsigned)((address >> 22) & 15);
            if (candidate_index != trigger_index && candidate_index != sibling &&
                (region_mask & (1U << region))) {
                UVM_ASSERT(sparse_count < ARRAY_SIZE(sparse_indices));
                sparse_indices[sparse_count++] = candidate_index;
            }
        }
        chunk_block_count = min(chunk_block_count, sparse_count);
    }

    // Reserve the entire bounded chunk before any speculative submission. The
    // reservation stays live through the final GPU wait on every exit path.
    reserved_bytes = uvm_pmm_gpu_eager_eviction_prefetch_begin(&gpu->pmm,
                                                               (NvU64)chunk_block_count * UVM_VA_BLOCK_SIZE,
                                                               &reservation);
    if (!reserved_bytes) {
        atomic64_inc(&uvm_cpu_preferred_prefetch_capacity_stops);
        uvm_processor_mask_set_atomic(&trigger_block->gpu_prefetch_retry, gpu->id);
        goto out;
    }
    prefetch_admitted = true;
    // A successful partial population must not make the whole VA block
    // one-shot. Later reports can name different CPU-resident pages in it.
    rearm_after_completion = limit_to_indicated;

    // Capacity denials are common under oversubscription. Obtain the byte
    // reservation before allocating a migration context so denied or already
    // claimed attempts need no context allocation/free. On allocation failure
    // release the reservation through the normal completion path and allow a
    // subsequent signal to try again.
    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context) {
        prefetch_success = false;
        rearm_after_completion = true;
        goto out;
    }

    atomic64_inc(&uvm_cpu_preferred_range_prefetches);
    if (retry)
        atomic64_inc(&uvm_cpu_preferred_prefetch_retries);

    chunk_block_count = (size_t)(reserved_bytes / UVM_VA_BLOCK_SIZE);
    UVM_ASSERT(chunk_block_count > 0 && chunk_block_count <= block_count);

    // Center the configurable chunk on the 2MB block which produced the
    // access-counter notification. Shift at allocation boundaries so a full
    // chunk is still considered whenever the range is large enough.
    start_index = trigger_index > chunk_block_count / 2 ?
                      trigger_index - chunk_block_count / 2 :
                      0;
    if (start_index + chunk_block_count > block_count)
        start_index = block_count - chunk_block_count;
    end_index = start_index + chunk_block_count;

    // Migrate the block which produced the notification before considering
    // its speculative neighbors. Otherwise, walking from start_index could
    // consume the available prefetch headroom and stop before reaching the
    // block which actually triggered this operation.
    for (block_index = 0; block_index < chunk_block_count; ++block_index) {
        uvm_va_block_t *va_block;
        NV_STATUS status;
        bool capacity_stop;
        NvU64 bytes_before;

        if (block_index > 0 && uvm_cpu_block_policy_counter_pressure(gpu)) {
            capacity_stopped = true;
            break;
        }

        if (refault_mode || region_mask) {
            candidate_index = sparse_indices[block_index];
        }
        else if (block_index == 0) {
            candidate_index = trigger_index;
        }
        else {
            candidate_index = start_index + block_index - 1;
            if (candidate_index >= trigger_index)
                ++candidate_index;
        }

        UVM_ASSERT(candidate_index < block_count);
        UVM_ASSERT(refault_mode || region_mask || (candidate_index >= start_index && candidate_index < end_index));

        va_block = candidate_index == trigger_index ?
                       trigger_block :
                       uvm_va_range_block(managed_range, candidate_index);

        // Speculative prefetch must not consume the headroom reserved by the
        // eager eviction policy. Demand migration can still make forward
        // progress while the background worker reclaims cold regions.
        if (uvm_pmm_gpu_eager_eviction_watermark_reached(&gpu->pmm)) {
            atomic64_inc(&uvm_cpu_preferred_prefetch_capacity_stops);
            capacity_stopped = true;
            uvm_pmm_gpu_prefetch_pressure_pause(&gpu->pmm);
            break;
        }

        // Managed pages are populated lazily. Avoid instantiating untouched VA
        // blocks just to create speculative zero-filled backing.
        if (!va_block)
            continue;

        bytes_before = prefetched_bytes;
        status = prefetch_block(va_block,
                                gpu,
                                mm,
                                block_context,
                                &prefetch_tracker,
                                &prefetched_bytes,
                                &reservation,
                                &capacity_stop,
                                candidate_index != trigger_index,
                                candidate_index == trigger_index ? indicated_pages : NULL,
                                candidate_index == trigger_index && limit_to_indicated);
        if (candidate_index == trigger_index)
            trigger_prefetched_bytes += prefetched_bytes - bytes_before;
        if (capacity_stop) {
            atomic64_inc(&uvm_cpu_preferred_prefetch_capacity_stops);
            capacity_stopped = true;
            prefetch_success = false;
            uvm_pmm_gpu_prefetch_allocation_failure(&gpu->pmm);
            break;
        }

        if (status != NV_OK) {
            prefetch_success = false;
            UVM_DBG_PRINT("Managed allocation prefetch stopped with %s on GPU %s\n",
                          nvstatusToString(status),
                          gpu->parent->name);
            break;
        }
    }

    atomic64_add(prefetched_bytes, &uvm_cpu_preferred_prefetched_bytes);
    atomic64_add(trigger_prefetched_bytes, &uvm_cpu_preferred_trigger_prefetched_bytes);
    atomic64_add(prefetched_bytes - trigger_prefetched_bytes, &uvm_cpu_preferred_neighbor_prefetched_bytes);

    // Keep completed attempts one-shot, but re-arm capacity-stopped attempts.
    // The started bit remains set so signals are cheap while usage is still
    // in watermark recovery or an eviction batch. Once recovery reaches the
    // low watermark and eviction finishes, one service thread
    // claims gpu_prefetch_retry and starts the next attempt.
    if (capacity_stopped)
        uvm_processor_mask_set_atomic(&trigger_block->gpu_prefetch_retry, gpu->id);

out:
    // No VA-block lock is held here. Drain all submitted work before allowing
    // eager eviction, even when submission stopped for capacity or an error.
    {
        NV_STATUS status = uvm_tracker_wait_deinit(&prefetch_tracker);
        if (status != NV_OK) {
            prefetch_success = false;
            UVM_DBG_PRINT("Managed allocation prefetch completion failed with %s on GPU %s\n",
                          nvstatusToString(status),
                          gpu->parent->name);
        }
    }

    if (prefetch_admitted)
        uvm_pmm_gpu_eager_eviction_prefetch_end(&gpu->pmm, &reservation,
                                                prefetched_bytes, prefetch_success);

    // Keep ownership through reservation release: another signal must not
    // claim this trigger while the failed attempt still owns capacity.
    if (rearm_after_completion && !capacity_stopped) {
        if (limit_to_indicated)
            uvm_processor_mask_clear_atomic(&trigger_block->gpu_prefetch_retry, gpu->id);
        uvm_processor_mask_clear_atomic(&trigger_block->gpu_prefetch_started, gpu->id);
    }

    if (block_context)
        uvm_va_block_context_free(block_context);
}

void uvm_cpu_block_policy_rearm_on_eviction(uvm_va_block_t *va_block, uvm_gpu_t *gpu, bool resident_evicted)
{
    uvm_va_block_gpu_state_t *gpu_state;
    NvU64 now = NV_GETTIME();

    uvm_assert_mutex_locked(&va_block->lock);

    gpu_state = uvm_va_block_gpu_state_get(va_block, gpu->id);
    if (gpu_state)
        gpu_state->counter_admission.pending = false;
    // This hook runs only after a successful eviction with pages removed.
    // Partial evictions deliberately do not create a new history episode.
    if (gpu_state && (uvm_perf_access_counter_admission_eviction_enable ||
                      uvm_perf_access_counter_admission_distance_stats_enable) &&
        va_block->managed_range && va_block->managed_range->cpu_access_counter_policy &&
        resident_evicted && uvm_page_mask_empty(&gpu_state->resident)) {
        gpu_state->counter_admission_history.evicted = true;
        gpu_state->counter_admission_history.population_at_eviction =
            (NvU64)atomic64_read(&gpu->counter_admission_population_bytes);
    }
    if (gpu_state && va_block->managed_range && va_block->managed_range->cpu_access_counter_policy)
        uvm_pmm_gpu_refault_evict(&gpu->pmm, &gpu_state->refault);
    if (gpu_state && gpu_state->speculative_prefetch_ns) {
        // Preserve the first eviction marker across further partial evictions.
        gpu_state->speculative_evicted_ns = 0;
        if (now >= gpu_state->speculative_prefetch_ns &&
            now - gpu_state->speculative_prefetch_ns <= uvm_pmm_gpu_prefetch_feedback_window_ns())
            gpu_state->speculative_evicted_ns = now;
        else
            gpu_state->speculative_cohort = 0;
        gpu_state->speculative_prefetch_ns = 0;
    }

    // A completed prefetch is one-shot only while its trigger block remains
    // GPU-resident. Once any part of that block is evicted to the CPU, a new
    // remote-access notification is genuine new demand and may start another
    // chunk prefetch. Also discard capacity-retry state: the eviction itself
    // supplies headroom and the next notification should be a fresh attempt.
    uvm_processor_mask_clear_atomic(&va_block->gpu_prefetch_retry, gpu->id);
    uvm_processor_mask_clear_atomic(&va_block->gpu_prefetch_started, gpu->id);
}

void uvm_cpu_block_policy_record_promotion(void)
{
    atomic64_inc(&uvm_cpu_preferred_block_promotions);
}
