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
#include "uvm_va_block.h"
#include "uvm_va_range.h"
#include "uvm_va_space.h"

static unsigned uvm_cpu_preferred_blocks_enable __read_mostly;
static atomic64_t uvm_cpu_preferred_block_promotions = ATOMIC64_INIT(0);

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

module_param(uvm_cpu_preferred_blocks_enable, uint, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_blocks_enable,
                 "Keep managed allocations CPU-preferred and promote complete populated VA blocks on GPU access-counter notifications.");
module_param_cb(uvm_cpu_preferred_block_promotions, &promotion_count_ops, NULL, S_IRUGO);
MODULE_PARM_DESC(uvm_cpu_preferred_block_promotions,
                 "Number of successful access-counter VA-block promotion service operations.");

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

bool uvm_cpu_block_policy_should_promote(uvm_va_block_t *va_block, uvm_gpu_t *gpu)
{
    uvm_va_range_managed_t *managed_range;

    if (!gpu || uvm_va_block_is_hmm(va_block))
        return false;

    uvm_assert_mutex_locked(&va_block->lock);
    managed_range = va_block->managed_range;
    if (!managed_range || !managed_range->cpu_access_counter_policy)
        return false;

    return uvm_cpu_block_policy_should_service_4k(managed_range, gpu);
}

void uvm_cpu_block_policy_record_promotion(void)
{
    atomic64_inc(&uvm_cpu_preferred_block_promotions);
}
