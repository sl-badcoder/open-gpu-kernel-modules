#!/usr/bin/env python3
"""Execute production observation, telemetry and lifecycle code with mocked Linux/RM."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).parents[2]
PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
typedef unsigned long long NvU64; typedef unsigned NvU32; typedef unsigned NV_STATUS;
#define NV_OK 0
#define NV_ERR_INVALID_STATE 0x40
#define NV_ERR_IN_USE 0x17
#define NSEC_PER_MSEC 1000000ULL
#define S_IRUGO 0
#define S_IWUSR 0
#define module_param(...)
#define MODULE_PARM_DESC(...)
#define READ_ONCE(x) (x)
#define WRITE_ONCE(x,v) ((x)=(v))
#define ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))
#define BUILD_BUG_ON(x) _Static_assert(!(x),"array mismatch")
#define hweight64(v) __builtin_popcountll(v)
#define clamp(v,l,h) ((v)<(l)?(l):((v)>(h)?(h):(v)))
#define min(a,b) ((a)<(b)?(a):(b))
#define UVM_ENTRY_VOID(x) do { x; } while(0)
#define container_of(p,t,m) ((t *)((char *)(p)-offsetof(t,m)))
static void info(const char *f,...) __attribute__((format(printf,1,2)));
static void info(const char *f,...) {(void)f;}
#define pr_info(...) info(__VA_ARGS__)
struct seq_file { char text[8192]; size_t used; };
static void seq_printf(struct seq_file *s,const char *f,...) __attribute__((format(printf,2,3)));
static void seq_printf(struct seq_file *s,const char *f,...) {
 va_list a;va_start(a,f);int n=vsnprintf(s->text+s->used,sizeof(s->text)-s->used,f,a);va_end(a);
 assert(n>=0 && (size_t)n<sizeof(s->text)-s->used);s->used+=(size_t)n;
}
static void seq_puts(struct seq_file *s,const char *text){seq_printf(s,"%s",text);}
struct mutex { int held; };
static bool disable_on_lock;
static void disable_sampling(void);
static void mutex_init(struct mutex *m){m->held=0;}
static void mutex_lock(struct mutex *m){assert(!m->held);m->held=1;if(disable_on_lock){disable_on_lock=false;disable_sampling();}}
static void mutex_unlock(struct mutex *m){assert(m->held);m->held=0;}
#define spin_lock_init(p) (*(p)=0)
#define spin_lock_irqsave(p,f) do {assert(!*(p));*(p)=1;(f)=0;}while(0)
#define spin_unlock_irqrestore(p,f) do {(void)(f);assert(*(p));*(p)=0;}while(0)
struct work_struct {int unused;};
struct delayed_work {struct work_struct work;void (*fn)(struct work_struct *);bool pending;};
#define to_delayed_work(w) container_of(w,struct delayed_work,work)
#define INIT_DELAYED_WORK(w,f) do {(w)->fn=(f);(w)->pending=false;}while(0)
static unsigned schedules,cancels;
static void schedule_delayed_work(struct delayed_work *w,unsigned delay){assert(delay>=5&&delay<=1000);++schedules;w->pending=true;}
static void cancel_delayed_work_sync(struct delayed_work *w){++cancels;w->pending=false;}
#define msecs_to_jiffies(x) (x)
static NvU64 now=1000000000ULL;
static NvU64 NV_GETTIME(void){return now;}
static struct {struct {int lock;}pm;} g_uvm_global;
static bool pm_allowed=true;static int pm_held;
static bool uvm_down_read_trylock(int *p){(void)p;if(!pm_allowed)return false;assert(!pm_held);pm_held=1;return true;}
static void uvm_up_read(int *p){(void)p;assert(pm_held);pm_held=0;}
'''
MOCKS = r'''
typedef struct {unsigned accessBitsBufferHandle;bool bDirtyBits;unsigned granularity;NvU64 enabledMask[64],currentBits[64];} UvmGpuAccessBitsBufferAlloc;
typedef unsigned UVM_ACCESS_BITS_DUMP_MODE;
#define UVM_ACCESS_BITS_DUMP_MODE_CURRENT 2
typedef struct parent {
 bool access_bits_supported;struct {bool enabled;}smc;void *rm_device;UvmGpuAccessBitsBufferAlloc vab_info;
 struct {struct mutex dump_lock;int cache_lock;struct delayed_work work;uvm_vab_snapshot_t sample;
 NvU64 samples,dump_errors,pm_skips,rising_bits,falling_bits,unchanged_samples,dump_duration_ns;
 NvU32 enabled_bits,set_bits;NV_STATUS last_dump_status;bool allocated,supported,stopping,running;}vab;
} uvm_parent_gpu_t;
typedef struct {int dummy;}uvm_pmm_gpu_t;
typedef struct {uvm_parent_gpu_t *parent;uvm_pmm_gpu_t pmm;}uvm_gpu_t;
static uvm_parent_gpu_t *active_parent;
static unsigned dumps;static NV_STATUS dump_status;static NvU64 dump_bits=1;
static NV_STATUS nvUvmInterfaceAccessBitsDump(void *dev,UvmGpuAccessBitsBufferAlloc *r,unsigned mode){
 (void)mode;uvm_parent_gpu_t *p=dev;assert(pm_held&&p->vab.dump_lock.held&&!p->vab.cache_lock);++dumps;
 now+=200000; if(!dump_status)r->currentBits[0]=dump_bits;return dump_status;
}
static void uvm_pmm_gpu_vab_stats(uvm_pmm_gpu_t *p,const uvm_vab_snapshot_t *s,uvm_pmm_gpu_vab_stats_t *out){
 (void)p;assert(!active_parent->vab.cache_lock);memset(out,0,sizeof(*out));
 for(unsigned i=0;i<2;++i){uvm_vab_region_state_t state=uvm_vab_classify(s,(NvU64)i*(4ULL<<20),2ULL<<20,0);
 if(state==UVM_VAB_REGION_HOT)++out->used_hot;else if(state==UVM_VAB_REGION_COLD)++out->used_cold;else ++out->used_unknown;}
}
'''
CASES = r'''
static void disable_sampling(void){uvm_perf_vab_mode=0;}
int main(void){
 uvm_parent_gpu_t p={0};uvm_vab_snapshot_t s;UvmGpuAccessBitsBufferAlloc raw;
 active_parent=&p;p.rm_device=&p;uvm_gpu_vab_init(&p);
 assert(uvm_perf_vab_mode==1 && uvm_perf_vab_eviction_linear_base==~0ULL);
 uvm_gpu_vab_get_snapshot(&p,&s);assert(!s.valid);
 p.access_bits_supported=true;p.vab_info.accessBitsBufferHandle=1;p.vab_info.granularity=6;
 uvm_gpu_vab_start(&p);assert(!schedules); // Empty mask must not start a worker.
 p.vab_info.enabledMask[0]=~0ULL;p.smc.enabled=true;uvm_gpu_vab_start(&p);assert(!schedules);
 p.smc.enabled=false;p.vab_info.bDirtyBits=true;uvm_gpu_vab_start(&p);assert(!schedules);
 p.vab_info.bDirtyBits=false;uvm_gpu_vab_start(&p);assert(schedules==1&&p.vab.running);
 NvU64 started=now;p.vab.work.fn(&p.vab.work.work);
 assert(dumps==1&&p.vab.samples==1&&p.vab.set_bits==1&&p.vab.enabled_bits==64);
 assert(p.vab.dump_duration_ns==200000&&p.vab.sample.sampled_ns==started);
 uvm_gpu_vab_get_snapshot(&p,&s);assert(s.valid);
 assert(uvm_vab_classify(&s,0,2ULL<<20,0)==UVM_VAB_REGION_UNKNOWN); // No base guessed.
 unsigned count=dumps;assert(uvm_gpu_vab_test_dump(&p,1,&raw)==NV_ERR_IN_USE&&dumps==count);
 dump_bits=3;vab_sample(&p);assert(p.vab.rising_bits==1&&!p.vab.falling_bits);
 dump_bits=2;vab_sample(&p);assert(p.vab.falling_bits==1&&p.vab.set_bits==1);
 vab_sample(&p);assert(p.vab.unchanged_samples==1);
 uvm_perf_vab_eviction_linear_base=0;uvm_gpu_vab_get_snapshot(&p,&s);
 assert(s.valid);
 assert(uvm_vab_classify(&s,0,2ULL<<20,0)==UVM_VAB_REGION_COLD);
 assert(uvm_vab_classify(&s,4ULL<<20,2ULL<<20,0)==UVM_VAB_REGION_HOT);
 struct seq_file report={0};uvm_gpu_t gpu={.parent=&p};count=dumps;
 uvm_gpu_vab_print(&gpu,&report);
 assert(dumps==count&&strstr(report.text,"mode 1\n")&&strstr(report.text,"selection_active 0\n"));
 assert(strstr(report.text,"granularity_kib 4096\n")&&strstr(report.text,"used_roots_cold 1\n"));
 assert(strstr(report.text,"used_roots_hot 1\n")&&strstr(report.text,"rising_bits 1\n"));
 uvm_perf_vab_mode=2;uvm_gpu_vab_get_snapshot(&p,&s);assert(!s.valid); // Unsupported mode cannot change policy.
 uvm_perf_vab_mode=1;uvm_perf_vab_eviction_linear_base=~0ULL;uvm_gpu_vab_get_snapshot(&p,&s);assert(s.valid);
 uvm_perf_vab_mode=0;count=dumps;vab_sample(&p);assert(dumps==count);
 uvm_gpu_vab_get_snapshot(&p,&s);assert(!s.valid);
 report=(struct seq_file){0};uvm_gpu_vab_print(&gpu,&report);
 assert(strstr(report.text,"worker_running 1\n")&&strstr(report.text,"sampling_active 0\n"));
 assert(uvm_down_read_trylock(&g_uvm_global.pm.lock));
 assert(uvm_gpu_vab_test_dump(&p,1,&raw)==NV_OK);uvm_up_read(&g_uvm_global.pm.lock);
 uvm_perf_vab_mode=1;vab_sample(&p);uvm_gpu_vab_get_snapshot(&p,&s);assert(s.valid);
 count=dumps;disable_on_lock=true;vab_sample(&p);
 assert(dumps==count&&!pm_held&&!p.vab.dump_lock.held);
 uvm_gpu_vab_get_snapshot(&p,&s);assert(!s.valid);
 uvm_perf_vab_mode=1;vab_sample(&p);
 uvm_gpu_vab_invalidate(&p);uvm_gpu_vab_get_snapshot(&p,&s);assert(!s.valid);
 vab_sample(&p);uvm_gpu_vab_get_snapshot(&p,&s);assert(s.valid);
 now+=3ULL*100*NSEC_PER_MSEC+1;uvm_gpu_vab_get_snapshot(&p,&s);assert(!s.valid);
 vab_sample(&p);pm_allowed=false;count=dumps;vab_sample(&p);
 assert(dumps==count&&p.vab.pm_skips==1);uvm_gpu_vab_get_snapshot(&p,&s);assert(!s.valid);
 pm_allowed=true;dump_status=NV_ERR_INVALID_STATE;vab_sample(&p);
 assert(p.vab.dump_errors==1&&p.vab.last_dump_status==dump_status);
 uvm_gpu_vab_get_snapshot(&p,&s);assert(!s.valid);
 dump_status=NV_OK;vab_sample(&p);uvm_gpu_vab_stop(&p);
 assert(cancels==1&&!p.vab.running&&!p.vab.work.pending);
 uvm_gpu_vab_get_snapshot(&p,&s);assert(!s.valid);
 count=dumps;unsigned scheduled=schedules;p.vab.work.fn(&p.vab.work.work);
 assert(dumps==count&&schedules==scheduled);
 uvm_perf_vab_mode=3;assert(!vab_mode());
 uvm_perf_vab_interval_ms=0;assert(vab_interval_ms()==5);
 uvm_perf_vab_interval_ms=2000;assert(vab_interval_ms()==1000);
 puts("VAB production observation, transitions, mode isolation, telemetry and lifecycle checks passed.");
 return 0;
}
'''

def main():
    header=(ROOT/'kernel-open/nvidia-uvm/uvm_vab.h').read_text().replace('#include "nvtypes.h"','')
    pmm_header=(ROOT/'kernel-open/nvidia-uvm/uvm_pmm_gpu.h').read_text()
    end=pmm_header.index('} uvm_pmm_gpu_vab_stats_t;')+len('} uvm_pmm_gpu_vab_stats_t;')
    start=pmm_header.rfind('typedef struct',0,end)
    stats=pmm_header[start:end]
    source=(ROOT/'kernel-open/nvidia-uvm/uvm_gpu_vab.c').read_text()
    source='\n'.join(line for line in source.splitlines() if not line.startswith('#include'))
    with tempfile.TemporaryDirectory(prefix='vab-observation-') as directory:
        c=Path(directory)/'test.c';exe=Path(directory)/'test'
        c.write_text(PRELUDE+header+stats+MOCKS+source+CASES)
        subprocess.run(['cc','-std=c11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(c),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)

if __name__=='__main__':
    main()
