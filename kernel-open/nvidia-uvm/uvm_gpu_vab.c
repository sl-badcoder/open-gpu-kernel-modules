// SPDX-License-Identifier: MIT
// Fully covered zero access bits classify a physical root as cold.
#include "uvm_gpu.h"
#include "uvm_api.h"
#include "uvm_global.h"
#include "uvm_vab.h"
#include "nv_uvm_interface.h"
#include <linux/bitops.h>
#include <linux/ktime.h>

#define UVM_VAB_SAMPLE_INTERVAL_MS 25
#define UVM_VAB_SAMPLE_MAX_AGE_NS (100ULL * NSEC_PER_MSEC)

// The legacy RM interface returns granularity but no range starts. Supplying
// this parameter explicitly confirms a single linear physical window on the
// target machine. Never guess a mapping or overwrite GSP-owned MMU registers.
static unsigned long long uvm_perf_vab_eviction_linear_base = ~0ULL;
module_param(uvm_perf_vab_eviction_linear_base, ullong, S_IRUGO);
MODULE_PARM_DESC(uvm_perf_vab_eviction_linear_base,
                "Assert verified single-linear-window VAB bit mapping with this physical base; unset disables VAB eviction");

static bool vab_eviction_supported(uvm_parent_gpu_t *parent_gpu)
{
    return uvm_perf_vab_eviction_linear_base != ~0ULL &&
           parent_gpu->access_bits_supported &&
           !parent_gpu->smc.enabled &&
           parent_gpu->vab_info.accessBitsBufferHandle &&
           !parent_gpu->vab_info.bDirtyBits &&
           parent_gpu->vab_info.granularity <= 15;
}

NV_STATUS uvm_gpu_vab_dump(uvm_parent_gpu_t *parent_gpu,
                           UVM_ACCESS_BITS_DUMP_MODE mode,
                           UvmGpuAccessBitsBufferAlloc *snapshot)
{
    NV_STATUS status;

    // Serialize with the test ioctl too: RM writes vab_info.currentBits.
    mutex_lock(&parent_gpu->vab_eviction.dump_lock);
    status = nvUvmInterfaceAccessBitsDump(parent_gpu->rm_device, &parent_gpu->vab_info, mode);
    if (status == NV_OK)
        *snapshot = parent_gpu->vab_info;
    mutex_unlock(&parent_gpu->vab_eviction.dump_lock);
    return status;
}

static void vab_sample(uvm_parent_gpu_t *parent_gpu)
{
    UvmGpuAccessBitsBufferAlloc snapshot;
    NV_STATUS status = NV_ERR_INVALID_STATE;
    unsigned long flags;

    // A trylock also prevents teardown (which can hold PM/global locks) from
    // waiting for a worker which is itself blocked on suspend.
    if (uvm_down_read_trylock(&g_uvm_global.pm.lock)) {
        status = uvm_gpu_vab_dump(parent_gpu, UVM_ACCESS_BITS_DUMP_MODE_CURRENT, &snapshot);
        uvm_up_read(&g_uvm_global.pm.lock);
    }

    spin_lock_irqsave(&parent_gpu->vab_eviction.cache_lock, flags);
    if (status == NV_OK) {
        parent_gpu->vab_eviction.sample = snapshot;
        parent_gpu->vab_eviction.sampled_ns = ktime_get_ns();
    }
    else {
        // Failed/PM-skipped observations never leave an old sample fresh.
        parent_gpu->vab_eviction.sampled_ns = 0;
    }
    spin_unlock_irqrestore(&parent_gpu->vab_eviction.cache_lock, flags);
}

static void vab_sample_work(struct work_struct *work)
{
    uvm_parent_gpu_t *parent_gpu = container_of(to_delayed_work(work),
                                               uvm_parent_gpu_t,
                                               vab_eviction.work);

    if (READ_ONCE(parent_gpu->vab_eviction.stopping))
        return;

    UVM_ENTRY_VOID(vab_sample(parent_gpu));
    if (!READ_ONCE(parent_gpu->vab_eviction.stopping))
        schedule_delayed_work(&parent_gpu->vab_eviction.work, msecs_to_jiffies(UVM_VAB_SAMPLE_INTERVAL_MS));
}

void uvm_gpu_vab_init(uvm_parent_gpu_t *parent_gpu)
{
    mutex_init(&parent_gpu->vab_eviction.dump_lock);
    spin_lock_init(&parent_gpu->vab_eviction.cache_lock);
    INIT_DELAYED_WORK(&parent_gpu->vab_eviction.work, vab_sample_work);
    parent_gpu->vab_eviction.stopping = true;
}

void uvm_gpu_vab_start(uvm_parent_gpu_t *parent_gpu)
{
    unsigned enabled_bits = 0;
    size_t word;

    if (uvm_perf_vab_eviction_linear_base == ~0ULL)
        return;

    if (!vab_eviction_supported(parent_gpu)) {
        pr_info("nvidia-uvm: VAB eviction inactive: unsupported device, SMC, dirty tracking, handle or granularity\n");
        return;
    }

    for (word = 0; word < ARRAY_SIZE(parent_gpu->vab_info.enabledMask); ++word)
        enabled_bits += hweight64(parent_gpu->vab_info.enabledMask[word]);
    if (!enabled_bits) {
        pr_info("nvidia-uvm: VAB eviction inactive: RM enabled mask is empty\n");
        return;
    }

    pr_info("nvidia-uvm: VAB eviction: base=0x%llx granularity=%llu KiB enabled=%u/%zu bits nominal_window=%llu MiB\n",
            uvm_perf_vab_eviction_linear_base,
            64ULL << parent_gpu->vab_info.granularity,
            enabled_bits,
            ARRAY_SIZE(parent_gpu->vab_info.currentBits) * 64,
            (64ULL << parent_gpu->vab_info.granularity) * ARRAY_SIZE(parent_gpu->vab_info.currentBits) * 64 / 1024);
    WRITE_ONCE(parent_gpu->vab_eviction.stopping, false);
    schedule_delayed_work(&parent_gpu->vab_eviction.work, msecs_to_jiffies(UVM_VAB_SAMPLE_INTERVAL_MS));
}

void uvm_gpu_vab_stop(uvm_parent_gpu_t *parent_gpu)
{
    unsigned long flags;

    WRITE_ONCE(parent_gpu->vab_eviction.stopping, true);
    cancel_delayed_work_sync(&parent_gpu->vab_eviction.work);
    spin_lock_irqsave(&parent_gpu->vab_eviction.cache_lock, flags);
    parent_gpu->vab_eviction.sampled_ns = 0;
    spin_unlock_irqrestore(&parent_gpu->vab_eviction.cache_lock, flags);
}

bool uvm_gpu_vab_snapshot(uvm_parent_gpu_t *parent_gpu,
                           UvmGpuAccessBitsBufferAlloc *snapshot,
                           NvU64 *linear_base)
{
    unsigned long flags;
    bool valid = false;

    // Only cached memory is consulted here. PMA can call this with RM's API
    // lock held, so issuing a synchronous RM dump here would deadlock.
    spin_lock_irqsave(&parent_gpu->vab_eviction.cache_lock, flags);
    if (uvm_vab_sample_fresh(ktime_get_ns(),
                             parent_gpu->vab_eviction.sampled_ns,
                             UVM_VAB_SAMPLE_MAX_AGE_NS)) {
        *snapshot = parent_gpu->vab_eviction.sample;
        *linear_base = uvm_perf_vab_eviction_linear_base;
        valid = true;
    }
    spin_unlock_irqrestore(&parent_gpu->vab_eviction.cache_lock, flags);
    return valid;
}
