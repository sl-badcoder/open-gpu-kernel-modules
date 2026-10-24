"""Compile production coverage helpers and chunk orchestration with mocked GPU work."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]

HARNESS = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
typedef uint64_t NvU64;
typedef int NV_STATUS;
#define NV_OK 0
#define UVM_TRACKER_INIT() {0}
#define UVM_ASSERT assert
#define UVM_DBG_PRINT(...) ((void)0)
#define UVM_SIZE_1MB 1048576ULL
#define UVM_VA_BLOCK_SIZE (2*UVM_SIZE_1MB)
#define READ_ONCE(x) (x)
#define max_t(t,a,b) ((t)(a)>(t)(b)?(t)(a):(t)(b))
#define min(a,b) ((a)<(b)?(a):(b))
typedef struct { int pending; } uvm_tracker_t;
typedef struct { int lock; } uvm_va_space_t;
struct mock_gpu;
typedef struct { struct { uvm_va_space_t *va_space; } va_range; int cpu_prefetch_lock; struct {struct mock_gpu *gpu; size_t first,outer;} cpu_prefetch_windows[8]; } uvm_va_range_managed_t;
typedef struct { uvm_va_range_managed_t *managed_range; NvU64 start; int gpu_prefetch_started,gpu_prefetch_retry; } uvm_va_block_t;
typedef struct mock_gpu { int pmm,id; struct { const char *name; } *parent; } uvm_gpu_t;
struct mm_struct { int unused; };
typedef struct { int unused; } uvm_va_block_context_t;
static unsigned uvm_cpu_preferred_prefetch_chunk_mb=8;
static int uvm_cpu_preferred_range_prefetches,uvm_cpu_preferred_prefetch_retries;
static int uvm_cpu_preferred_prefetch_capacity_stops,uvm_cpu_preferred_prefetched_bytes;
static int submitted,waits,ends,admitted,mode,freed,allocations;
static uvm_va_space_t space;
static uvm_va_range_managed_t range={.va_range={&space}};
#define ARRAY_SIZE(x) (sizeof(x)/sizeof((x)[0]))
static void uvm_spin_lock(int *p){assert(!*p);*p=1;}
static void uvm_spin_unlock(int *p){assert(*p);*p=0;}
static uvm_va_block_t blocks[4];
static uvm_va_block_context_t ctx;
#define atomic64_inc(p) (++*(p))
#define atomic64_add(v,p) (*(p)+=(v))
static bool should_handle_gpu_signal(uvm_va_block_t *b,uvm_gpu_t *g) { (void)b;(void)g; return true; }
#define uvm_assert_rwsem_locked_read(p) ((void)(p))
static bool uvm_processor_mask_test(int *p,int id) { (void)id;return *p; }
static bool uvm_processor_mask_test_and_clear_atomic(int *p,int id) { bool v=*p;(void)id;*p=0;return v; }
static void uvm_processor_mask_clear_atomic(int *p,int id) { (void)id;*p=0; }
static bool uvm_processor_mask_test_and_set_atomic(int *p,int id) { bool v=*p;(void)id;*p=1;return v; }
static void uvm_processor_mask_set_atomic(int *p,int id) { (void)id;*p=1; }
static bool uvm_pmm_gpu_eager_eviction_watermark_reached(int *p) { (void)p;return mode==5 || (mode==1 && submitted==2); }
static bool uvm_pmm_gpu_eager_eviction_prefetch_begin(int *p) { (void)p;if(mode==3)return false;assert(!admitted);admitted=1;return true; }
static void uvm_pmm_gpu_eager_eviction_prefetch_end(int *p) { (void)p;assert(admitted && (waits==1 || !submitted));admitted=0;++ends; }
static uvm_va_block_context_t *uvm_va_block_context_alloc(struct mm_struct *m) { (void)m;++allocations;return &ctx; }
static void uvm_va_block_context_free(uvm_va_block_context_t *c) { (void)c;++freed; }
static size_t uvm_va_range_num_blocks(uvm_va_range_managed_t *r) { (void)r;return 4; }
static size_t uvm_va_range_block_index(uvm_va_range_managed_t *r,NvU64 a) { (void)r;return a/UVM_VA_BLOCK_SIZE; }
static uvm_va_block_t *uvm_va_range_block(uvm_va_range_managed_t *r,size_t i) { (void)r;return &blocks[i]; }
static NV_STATUS prefetch_block(uvm_va_block_t *b,uvm_gpu_t *g,struct mm_struct *m,uvm_va_block_context_t *c,uvm_tracker_t *t,NvU64 *bytes,bool *stop) {
(void)b;(void)g;(void)m;(void)c;assert(admitted && !waits);if(mode==7 && !submitted){assert(blocks[0].gpu_prefetch_retry);range.cpu_prefetch_windows[0].gpu=NULL;}++submitted;++t->pending;*bytes+=4096;*stop=false;return mode==2 && submitted==2 ? 1:NV_OK;
}
static NV_STATUS uvm_tracker_wait_deinit(uvm_tracker_t *t) { if(t->pending){assert(admitted);assert(submitted==((mode==0 || mode==9 || mode==10)?4:2));++waits;t->pending=0;} return NV_OK; }
'''

SCENARIOS = r'''
int main(void) {
    uvm_gpu_t gpu={0}, other_gpu={0};
    prefetch_coverage_t coverage[8];
    unsigned n;
    int slots[8];

    // Actual production helpers: GPU isolation, interval bounds, saturation,
    // non-overlap and complete cleanup. No persistent residency cache exists.
    for (int i=0;i<8;++i) {
        slots[i]=prefetch_window_begin(&range,&gpu,&blocks[0],i,i+4,coverage,&n);
        assert(slots[i]==i);
        for(unsigned j=0;j<n;++j) assert(coverage[j].first < (size_t)i+4 && coverage[j].outer > (size_t)i);
    }
    assert(prefetch_window_begin(&range,&gpu,&blocks[0],2,6,coverage,&n)==-1 && n>0);
    assert(prefetch_window_covers(coverage,n,3));
    assert(blocks[0].gpu_prefetch_retry);
    assert(!prefetch_window_covers(coverage,n,20));
    assert(prefetch_window_begin(&range,&other_gpu,&blocks[0],2,6,coverage,&n)==-1 && !n);
    for(int i=0;i<8;++i) prefetch_window_end(&range,slots[i]);
    assert(prefetch_window_begin(&range,&gpu,&blocks[0],100,110,coverage,&n)==0 && !n);
    prefetch_window_end(&range,0);
    prefetch_window_end(&range,-1);

    // Execute the actual chunk orchestration, with migration mocked only.
    for(mode=0;mode<=10;++mode) {
        if(mode==5 || mode==6) continue;
        submitted=waits=ends=admitted=freed=allocations=0;
        for(int i=0;i<4;++i) blocks[i]=(uvm_va_block_t){.managed_range=&range,.start=i*UVM_VA_BLOCK_SIZE};
        if(mode==4) blocks[0].gpu_prefetch_started=1;
        if(mode==7) {range.cpu_prefetch_windows[0].gpu=&gpu; range.cpu_prefetch_windows[0].first=1;range.cpu_prefetch_windows[0].outer=3;}
        if(mode==8) {range.cpu_prefetch_windows[0].gpu=&gpu; range.cpu_prefetch_windows[0].first=0;range.cpu_prefetch_windows[0].outer=4;}
        if(mode==9) {range.cpu_prefetch_windows[0].gpu=&other_gpu;range.cpu_prefetch_windows[0].first=0;range.cpu_prefetch_windows[0].outer=4;}
        if(mode==10) for(int i=0;i<8;++i){range.cpu_prefetch_windows[i].gpu=&other_gpu;range.cpu_prefetch_windows[i].first=0;range.cpu_prefetch_windows[i].outer=4;}
        uvm_cpu_block_policy_prefetch_on_signal(&blocks[0],&gpu,NULL);
        assert(!admitted && !range.cpu_prefetch_lock);
        if(mode<3 || mode==7 || mode==9 || mode==10) assert(waits==1 && ends==1);
        if(mode==1 || mode==3 || mode==7 || mode==8) assert(blocks[0].gpu_prefetch_retry);
        if(mode==8) assert(!submitted && !waits && ends==1);
        if(mode==3 || mode==4) assert(!submitted && !waits && !ends);
        if(mode==8 || mode==9) assert(range.cpu_prefetch_windows[0].gpu && !range.cpu_prefetch_windows[1].gpu);
        if(mode==7) assert(!range.cpu_prefetch_windows[0].gpu && !range.cpu_prefetch_windows[1].gpu);
        if(mode==10) for(int i=0;i<8;++i) assert(range.cpu_prefetch_windows[i].gpu==&other_gpu);
        for(int i=0;i<8;++i) range.cpu_prefetch_windows[i].gpu=NULL;
    }
    puts("Passed: overlap skipping/retry, GPU isolation, bounded saturation fallback, half-open bounds, cleanup and chunk completion/error paths.");
}
'''

def main():
    source = (ROOT / "kernel-open/nvidia-uvm/uvm_cpu_block_policy.c").read_text()
    first = source.index("typedef struct\n{\n    size_t first;")
    last = source.index("void uvm_cpu_block_policy_rearm_on_eviction", first)
    with tempfile.TemporaryDirectory(prefix="uvm-coalescing-") as directory:
        path = Path(directory)
        (path / "test.c").write_text(HARNESS + source[first:last] + SCENARIOS)
        subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", str(path / "test.c"), "-o", str(path / "test")], check=True)
        subprocess.run([str(path / "test")], check=True)

if __name__ == "__main__":
    main()
