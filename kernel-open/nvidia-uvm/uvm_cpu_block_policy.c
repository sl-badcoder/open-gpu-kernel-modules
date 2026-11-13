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

static unsigned uvm_cpu_preferred_blocks_enable __read_mostly = 1;
#define UVM_CPU_REGION_SIZE (2ULL * UVM_VA_BLOCK_SIZE)
static atomic64_t uvm_cpu_preferred_region_promotions = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_region_migrated_bytes = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_region_failures = ATOMIC64_INIT(0);

static int atomic64_count_get(char *buffer, const struct kernel_param *kp)
{
    atomic64_t *counter = kp->arg;

    return scnprintf(buffer, PAGE_SIZE, "%lld\n", (long long)atomic64_read(counter));
}

static const struct kernel_param_ops atomic64_count_ops = {
    .get = atomic64_count_get,
};

/**
 * Introduced module params + description
 * Define new module params here
 */
module_param_cb(uvm_cpu_preferred_region_promotions, &atomic64_count_ops, &uvm_cpu_preferred_region_promotions, S_IRUGO);
module_param_cb(uvm_cpu_preferred_region_migrated_bytes, &atomic64_count_ops, &uvm_cpu_preferred_region_migrated_bytes, S_IRUGO);
module_param_cb(uvm_cpu_preferred_region_failures, &atomic64_count_ops, &uvm_cpu_preferred_region_failures, S_IRUGO);
module_param(uvm_cpu_preferred_blocks_enable, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_blocks_enable, "Keep managed allocations CPU-preferred and enable access-counter-guided migration policy.");


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

    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (!managed_range->cpu_access_counter_policy ||
        !UVM_ID_IS_CPU(managed_range->policy.preferred_location) ||
        !gpu_supported(va_space, gpu)) {
        return false;
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

bool uvm_cpu_block_policy_should_migrate_region(uvm_va_block_t *va_block, uvm_gpu_t *gpu)
{
    return should_handle_gpu_signal(va_block, gpu);
}

// The page mask persists across allocation and mapping retries: make_resident
// can have populated pages before map asks to retry. Never recompute this mask
// from CPU residency after a partial successful transition.
static NV_STATUS migrate_region_block_locked(uvm_va_block_t *va_block,
                                             uvm_gpu_t *gpu,
                                             uvm_va_block_retry_t *retry,
                                             uvm_va_block_context_t *context,
                                             const uvm_page_mask_t *pages,
                                             uvm_tracker_t *tracker,
                                             NvU64 *moved_bytes,
                                             bool require_mapping,
                                             bool *started,
                                             bool *stale)
{
    NV_STATUS status, tracker_status;
    uvm_assert_mutex_locked(&va_block->lock);
    // Reject old notifications before taking ownership of the transition.
    // make_resident itself unmaps remote GPU PTEs before allocating/copying;
    // after that, a cleared mapped bit is expected on allocation/map retries.
    if (require_mapping && !*started && !uvm_processor_mask_test(&va_block->mapped, gpu->id)) {
        *stale = true;
        return NV_OK;
    }
    *started = true;
    uvm_page_mask_zero(&context->make_resident.pages_changed_residency);
    uvm_processor_mask_zero(&context->make_resident.all_involved_processors);
    status = uvm_va_block_make_resident(va_block, retry, context, gpu->id,
                                        uvm_va_block_region_from_block(va_block), pages, NULL,
                                        UVM_MAKE_RESIDENT_CAUSE_ACCESS_COUNTER);
    *moved_bytes += (NvU64)uvm_page_mask_weight(&context->make_resident.pages_changed_residency) * PAGE_SIZE;
    if (status == NV_OK)
        status = uvm_va_block_map(va_block, context, gpu->id,
                                  uvm_va_block_region_from_block(va_block), pages,
                                  UVM_PROT_READ_WRITE_ATOMIC, UvmEventMapRemoteCauseInvalid,
                                  &va_block->tracker);
    // Preserve dependencies from successful and partial submissions on errors.
    tracker_status = uvm_tracker_add_tracker_safe(tracker, &va_block->tracker);
    if ((status == NV_OK || status == NV_ERR_MORE_PROCESSING_REQUIRED) && tracker_status != NV_OK)
        status = tracker_status;
    return status;
}

NV_STATUS uvm_cpu_block_policy_migrate_region_on_signal(uvm_va_block_t *trigger_block,
                                                        uvm_gpu_t *gpu,
                                                        struct mm_struct *mm,
                                                        NvU32 buffer_index,
                                                        bool *did_migrate)
{
    uvm_va_range_managed_t *range = trigger_block->managed_range;
    uvm_va_block_context_t *context;
    uvm_tracker_t tracker = UVM_TRACKER_INIT();
    NvU64 base, first, last, moved_bytes = 0;
    size_t first_index, last_index, trigger_index, index, pass;
    NV_STATUS status = NV_OK, wait_status;
    bool stale = false;

    *did_migrate = false;
    if (!uvm_cpu_block_policy_should_migrate_region(trigger_block, gpu))
        return NV_OK;
    uvm_assert_rwsem_locked_read(&range->va_range.va_space->lock);

    // Absolute 4MiB VA alignment; never cross a managed allocation boundary.
    // PMM still assigns independent 2MiB physical roots to these virtual blocks.
    base = trigger_block->start & ~(UVM_CPU_REGION_SIZE - 1);
    first = max(base, range->va_range.node.start);
    last = min(base + UVM_CPU_REGION_SIZE - 1, range->va_range.node.end);
    first_index = uvm_va_range_block_index(range, first);
    last_index = uvm_va_range_block_index(range, last);
    trigger_index = uvm_va_range_block_index(range, trigger_block->start);
    UVM_ASSERT(last_index >= first_index && last_index - first_index < 2);
    // Match native access-counter service: a notification is stale once the
    // triggering GPU mapping has gone away. Check under its block lock again
    // after notification aggregation, before touching either region half.
    uvm_mutex_lock(&trigger_block->lock);
    if (!uvm_processor_mask_test(&trigger_block->mapped, gpu->id)) {
        uvm_mutex_unlock(&trigger_block->lock);
        return NV_OK;
    }
    uvm_mutex_unlock(&trigger_block->lock);
    context = uvm_va_block_context_alloc(mm);
    if (!context) {
        atomic64_inc(&uvm_cpu_preferred_region_failures);
        return NV_ERR_NO_MEMORY;
    }

    // Service the triggering half first. Hold only one VA-block lock at a time.
    for (pass = 0; pass <= last_index - first_index; ++pass) {
        uvm_va_block_t *block;
        uvm_va_block_retry_t retry;
        bool started = false;
        uvm_page_mask_t pages;
        const uvm_page_mask_t *cpu_resident, *gpu_resident, *thrashing;
        index = pass == 0 ? trigger_index : (trigger_index == first_index ? last_index : first_index);
        if (index != trigger_index) {
            uvm_mutex_lock(&trigger_block->lock);
            stale = !uvm_processor_mask_test(&trigger_block->mapped, gpu->id);
            uvm_mutex_unlock(&trigger_block->lock);
            if (stale)
                break;
        }
        block = index == trigger_index ? trigger_block : uvm_va_range_block(range, index);
        // Do not manufacture backing for untouched or unpopulated neighbors.
        if (!block)
            continue;
        uvm_mutex_lock(&block->lock);
        if (block == trigger_block && !uvm_processor_mask_test(&block->mapped, gpu->id)) {
            uvm_mutex_unlock(&block->lock);
            break;
        }
        cpu_resident = uvm_va_block_resident_mask_get(block, UVM_ID_CPU, NUMA_NO_NODE);
        if (!cpu_resident) {
            uvm_mutex_unlock(&block->lock);
            continue;
        }
        uvm_page_mask_copy(&pages, cpu_resident);
        gpu_resident = uvm_va_block_resident_mask_get(block, gpu->id, NUMA_NO_NODE);
        if (gpu_resident)
            uvm_page_mask_andnot(&pages, &pages, gpu_resident);
        thrashing = uvm_perf_thrashing_get_thrashing_pages(block);
        if (thrashing)
            uvm_page_mask_andnot(&pages, &pages, thrashing);
        if (!uvm_page_mask_empty(&pages)) {
            uvm_va_block_context_init(context, mm);
            context->make_resident.access_counters_buffer_index = buffer_index;
            status = UVM_VA_BLOCK_RETRY_LOCKED(block, &retry,
                         migrate_region_block_locked(block, gpu, &retry, context, &pages, &tracker, &moved_bytes,
                                                       block == trigger_block, &started, &stale));
        }
        uvm_mutex_unlock(&block->lock);
        if (status != NV_OK || stale)
            break;
    }
    // Drain both halves before acknowledging/clearing the threshold notification.
    wait_status = uvm_tracker_wait_deinit(&tracker);
    if (status == NV_OK)
        status = wait_status;
    *did_migrate = moved_bytes != 0;
    atomic64_add(moved_bytes, &uvm_cpu_preferred_region_migrated_bytes);
    if (status != NV_OK)
        atomic64_inc(&uvm_cpu_preferred_region_failures);
    else if (moved_bytes)
        atomic64_inc(&uvm_cpu_preferred_region_promotions);
    uvm_va_block_context_free(context);
    return status;
}
