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

#define UVM_CPU_PREFERRED_PREFETCH_CHUNK_MB_DEFAULT 64

static unsigned uvm_cpu_preferred_blocks_enable __read_mostly;
static unsigned uvm_cpu_preferred_block_promotion_enable __read_mostly = 1;
static unsigned uvm_cpu_preferred_prefetch_chunk_mb __read_mostly =
    UVM_CPU_PREFERRED_PREFETCH_CHUNK_MB_DEFAULT;
static atomic64_t uvm_cpu_preferred_block_promotions = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_range_prefetches = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_prefetched_bytes = ATOMIC64_INIT(0);
static atomic64_t uvm_cpu_preferred_prefetch_capacity_stops = ATOMIC64_INIT(0);

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

module_param(uvm_cpu_preferred_blocks_enable, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_blocks_enable,
                 "Keep managed allocations CPU-preferred and enable access-counter-guided migration policy.");
module_param(uvm_cpu_preferred_block_promotion_enable, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_cpu_preferred_block_promotion_enable,
                 "Promote complete populated 2MB VA blocks on GPU access-counter notifications. "
                 "Zero leaves speculative prefetch enabled without block promotion.");
module_param(uvm_cpu_preferred_prefetch_chunk_mb, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_cpu_preferred_prefetch_chunk_mb,
                 "Access-counter-guided prefetch chunk size in MiB. Non-zero values are rounded up to 2MB; "
                 "zero disables speculative chunk prefetch.");
module_param_cb(uvm_cpu_preferred_block_promotions, &promotion_count_ops, NULL, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_block_promotions,
                 "Number of successful access-counter VA-block promotion service operations.");
module_param_cb(uvm_cpu_preferred_range_prefetches,
                &atomic64_count_ops,
                &uvm_cpu_preferred_range_prefetches,
                S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_range_prefetches,
                 "Number of access-counter-guided managed allocation chunk prefetches started.");
module_param_cb(uvm_cpu_preferred_prefetched_bytes,
                &atomic64_count_ops,
                &uvm_cpu_preferred_prefetched_bytes,
                S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_prefetched_bytes,
                 "Number of bytes moved by access-counter-guided managed allocation chunk prefetches.");
module_param_cb(uvm_cpu_preferred_prefetch_capacity_stops,
                &atomic64_count_ops,
                &uvm_cpu_preferred_prefetch_capacity_stops,
                S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_prefetch_capacity_stops,
                 "Number of managed allocation prefetches stopped when free VRAM was exhausted.");

bool uvm_cpu_block_policy_enabled(void)
{
    return uvm_cpu_preferred_blocks_enable != 0;
}

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
    uvm_va_block_t *va_block;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (!managed_range->cpu_access_counter_policy ||
        !UVM_ID_IS_CPU(managed_range->policy.preferred_location) ||
        !gpu_supported(va_space, gpu)) {
        return false;
    }

    // A GPU may be unregistered and registered again in the same VA space.
    // Permit the newly registered instance to trigger each block again.
    for_each_va_block_in_va_range(managed_range, va_block)
        uvm_processor_mask_clear_atomic(&va_block->gpu_prefetch_started, gpu->id);

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
                                NvU64 *prefetched_bytes,
                                bool *capacity_stop)
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
    if (gpu_resident)
        uvm_page_mask_andnot(&prefetch_mask, &prefetch_mask, gpu_resident);

    // Do not speculatively move pages which UVM's thrashing policy is
    // currently protecting from migration.
    thrashing_pages = uvm_perf_thrashing_get_thrashing_pages(va_block);
    if (thrashing_pages)
        uvm_page_mask_andnot(&prefetch_mask, &prefetch_mask, thrashing_pages);

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
        *prefetched_bytes +=
            (NvU64)uvm_page_mask_weight(&block_context->make_resident.pages_changed_residency) * PAGE_SIZE;

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

    uvm_mutex_unlock(&va_block->lock);
    uvm_va_block_retry_deinit(&retry, va_block);
    return status;

out_unlock:
    uvm_mutex_unlock(&va_block->lock);
    return status;
}

void uvm_cpu_block_policy_prefetch_on_signal(uvm_va_block_t *trigger_block,
                                             uvm_gpu_t *gpu,
                                             struct mm_struct *mm)
{
    uvm_va_range_managed_t *managed_range = trigger_block->managed_range;
    uvm_va_space_t *va_space;
    uvm_va_block_context_t *block_context;
    size_t block_count;
    size_t chunk_block_count;
    size_t end_index;
    size_t start_index;
    size_t trigger_index;
    size_t block_index;
    NvU64 chunk_bytes;
    NvU64 prefetched_bytes = 0;
    unsigned prefetch_chunk_mb = READ_ONCE(uvm_cpu_preferred_prefetch_chunk_mb);

    if (!managed_range ||
        prefetch_chunk_mb == 0 ||
        !should_handle_gpu_signal(trigger_block, gpu)) {
        return;
    }

    va_space = managed_range->va_range.va_space;
    uvm_assert_rwsem_locked_read(&va_space->lock);

    if (uvm_processor_mask_test(&trigger_block->gpu_prefetch_started, gpu->id))
        return;

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return;

    if (uvm_processor_mask_test_and_set_atomic(&trigger_block->gpu_prefetch_started, gpu->id))
        goto out;

    atomic64_inc(&uvm_cpu_preferred_range_prefetches);

    block_count = uvm_va_range_num_blocks(managed_range);
    trigger_index = uvm_va_range_block_index(managed_range, trigger_block->start);
    chunk_bytes = (NvU64)prefetch_chunk_mb * UVM_SIZE_1MB;
    chunk_block_count = (chunk_bytes + UVM_VA_BLOCK_SIZE - 1) / UVM_VA_BLOCK_SIZE;
    chunk_block_count = max_t(size_t, chunk_block_count, 1);
    chunk_block_count = min(chunk_block_count, block_count);

    // Center the configurable chunk on the 2MB block which produced the
    // access-counter notification. Shift at allocation boundaries so a full
    // chunk is still considered whenever the range is large enough.
    start_index = trigger_index > chunk_block_count / 2 ?
                      trigger_index - chunk_block_count / 2 :
                      0;
    if (start_index + chunk_block_count > block_count)
        start_index = block_count - chunk_block_count;
    end_index = start_index + chunk_block_count;

    for (block_index = start_index; block_index < end_index; ++block_index) {
        uvm_va_block_t *va_block = uvm_va_range_block(managed_range, block_index);
        NV_STATUS status;
        bool capacity_stop;

        // Managed pages are populated lazily. Avoid instantiating untouched VA
        // blocks just to create speculative zero-filled backing.
        if (!va_block)
            continue;

        status = prefetch_block(va_block, gpu, mm, block_context, &prefetched_bytes, &capacity_stop);
        if (capacity_stop) {
            atomic64_inc(&uvm_cpu_preferred_prefetch_capacity_stops);
            break;
        }

        if (status != NV_OK) {
            UVM_DBG_PRINT("Managed allocation prefetch stopped with %s on GPU %s\n",
                          nvstatusToString(status),
                          gpu->parent->name);
            break;
        }
    }

    atomic64_add(prefetched_bytes, &uvm_cpu_preferred_prefetched_bytes);

out:
    uvm_va_block_context_free(block_context);
}

void uvm_cpu_block_policy_record_promotion(void)
{
    atomic64_inc(&uvm_cpu_preferred_block_promotions);
}
