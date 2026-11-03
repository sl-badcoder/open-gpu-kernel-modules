#!/usr/bin/env python3
"""Execute production sampler control flow with mocked Linux/RM primitives."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).parents[2]
PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
typedef uint64_t NvU64; typedef uint32_t NvU32; typedef int NV_STATUS;
#define NV_OK 0
#define NV_ERR_INVALID_STATE 1
#define NSEC_PER_MSEC 1000000ULL
#define S_IRUGO 0
#define module_param(...)
#define MODULE_PARM_DESC(...)
#define pr_info(...) ((void)0)
#define ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))
#define hweight64(v) __builtin_popcountll(v)
#define READ_ONCE(x) (x)
#define WRITE_ONCE(x,v) ((x)=(v))
#define UVM_ENTRY_VOID(x) do { x; } while(0)
#define container_of(p,t,m) ((t *)((char *)(p)-offsetof(t,m)))
struct work_struct { int unused; };
struct delayed_work { struct work_struct work; void (*fn)(struct work_struct *); int pending; };
#define to_delayed_work(w) container_of(w,struct delayed_work,work)
struct mutex { int held; };
static void mutex_init(struct mutex *m) { m->held=0; }
static void mutex_lock(struct mutex *m) { assert(!m->held); m->held=1; }
static void mutex_unlock(struct mutex *m) { assert(m->held); m->held=0; }
#define spin_lock_init(p) (*(p)=0)
#define spin_lock_irqsave(p,f) do { assert(!*(p)); *(p)=1; (f)=0; } while(0)
#define spin_unlock_irqrestore(p,f) do { (void)(f); assert(*(p)); *(p)=0; } while(0)
#define INIT_DELAYED_WORK(w,f) do { (w)->fn=(f); (w)->pending=0; } while(0)
static int scheduled,cancelled;
static void schedule_delayed_work(struct delayed_work *w,unsigned delay) { assert(delay==25); w->pending=1; ++scheduled; }
static void cancel_delayed_work_sync(struct delayed_work *w) { w->pending=0; ++cancelled; }
#define msecs_to_jiffies(x) (x)
typedef struct { int accessBitsBufferHandle; bool bDirtyBits; NvU32 granularity; NvU64 enabledMask[64],currentBits[64]; } UvmGpuAccessBitsBufferAlloc;
typedef int UVM_ACCESS_BITS_DUMP_MODE;
#define UVM_ACCESS_BITS_DUMP_MODE_CURRENT 2
typedef struct {
 bool access_bits_supported;
 struct { bool enabled; } smc;
 void *rm_device;
 UvmGpuAccessBitsBufferAlloc vab_info;
 struct { struct mutex dump_lock; int cache_lock; struct delayed_work work; UvmGpuAccessBitsBufferAlloc sample; NvU64 sampled_ns; bool stopping; } vab_eviction;
} uvm_parent_gpu_t;
static struct { struct { int lock; } pm; } g_uvm_global;
static bool pm_allowed=true; static int pm_held,dumps; static NV_STATUS dump_status;
static int uvm_down_read_trylock(int *p) { (void)p; if (!pm_allowed) return 0; assert(!pm_held); pm_held=1; return 1; }
static void uvm_up_read(int *p) { (void)p; assert(pm_held); pm_held=0; }
static NvU64 now=1000000000ULL;
static NvU64 ktime_get_ns(void) { return now; }
static NV_STATUS nvUvmInterfaceAccessBitsDump(void *device,UvmGpuAccessBitsBufferAlloc *info,UVM_ACCESS_BITS_DUMP_MODE mode) {
 (void)device; assert(pm_held && mode==UVM_ACCESS_BITS_DUMP_MODE_CURRENT); ++dumps;
 if (dump_status==NV_OK) info->currentBits[0]=5;
 return dump_status;
}
'''
TESTS = r'''
int main(void) {
 uvm_parent_gpu_t p={0}; UvmGpuAccessBitsBufferAlloc snap; NvU64 base;
 assert(!uvm_vab_region_cold(NULL,NULL,0,0,0,0,1));
 uvm_gpu_vab_init(&p);
 assert(!uvm_gpu_vab_snapshot(&p,&snap,&base));
 p.access_bits_supported=true; p.vab_info.accessBitsBufferHandle=1; p.vab_info.granularity=5;
 uvm_gpu_vab_start(&p); assert(!scheduled); // mapping unset
 uvm_perf_vab_eviction_linear_base=0;
 p.smc.enabled=true; uvm_gpu_vab_start(&p); assert(!scheduled);
 p.smc.enabled=false; p.vab_info.bDirtyBits=true; uvm_gpu_vab_start(&p); assert(!scheduled);
 p.vab_info.bDirtyBits=false; p.vab_info.granularity=16; uvm_gpu_vab_start(&p); assert(!scheduled);
 p.vab_info.granularity=5; uvm_gpu_vab_start(&p); assert(!scheduled); // empty mask
 p.vab_info.enabledMask[0]=~0ULL; uvm_gpu_vab_start(&p); assert(scheduled==1 && !p.vab_eviction.stopping);
 p.vab_eviction.work.fn(&p.vab_eviction.work.work);
 assert(dumps==1 && scheduled==2 && !pm_held && !p.vab_eviction.dump_lock.held);
 assert(uvm_gpu_vab_snapshot(&p,&snap,&base) && snap.currentBits[0]==5 && base==0);
 now+=UVM_VAB_SAMPLE_MAX_AGE_NS+1; assert(!uvm_gpu_vab_snapshot(&p,&snap,&base));
 dump_status=1; vab_sample(&p); assert(!uvm_gpu_vab_snapshot(&p,&snap,&base));
 dump_status=NV_OK; vab_sample(&p); assert(uvm_gpu_vab_snapshot(&p,&snap,&base));
 int before=dumps; pm_allowed=false; vab_sample(&p);
 assert(dumps==before && !uvm_gpu_vab_snapshot(&p,&snap,&base));
 pm_allowed=true; vab_sample(&p); assert(uvm_gpu_vab_snapshot(&p,&snap,&base));
 uvm_gpu_vab_stop(&p); assert(cancelled==1 && p.vab_eviction.stopping && !p.vab_eviction.work.pending);
 assert(!uvm_gpu_vab_snapshot(&p,&snap,&base));
 before=scheduled; int dumps_before=dumps;
 p.vab_eviction.work.fn(&p.vab_eviction.work.work); assert(scheduled==before && dumps==dumps_before);
 return 0;
}
'''

def main():
    helper = (ROOT / 'kernel-open/nvidia-uvm/uvm_vab.h').read_text().replace('#include "nvtypes.h"', '')
    sampler = (ROOT / 'kernel-open/nvidia-uvm/uvm_gpu_vab.c').read_text()
    sampler = '\n'.join(line for line in sampler.splitlines() if not line.startswith('#include'))
    with tempfile.TemporaryDirectory(prefix='uvm-vab-sampling-') as directory:
        c = Path(directory) / 'sample.c'
        out = Path(directory) / 'sample'
        c.write_text(PRELUDE + helper + sampler + TESTS)
        subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', str(c), '-o', str(out)], check=True)
        subprocess.run([str(out)], check=True)
    print('VAB production sampler lifecycle checks passed (mock Linux/RM).')

if __name__ == '__main__':
    main()
