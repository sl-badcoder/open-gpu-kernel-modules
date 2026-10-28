#!/usr/bin/env python3
"""Exercise the production PMA headroom accounting and prefetch admission."""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "kernel-open/nvidia-uvm/uvm_pmm_gpu.c"


def extract(source, name):
    match = re.search(r"^(?:static )?(?:bool|unsigned|void|NV_STATUS) [^\n]*\b" + re.escape(name) + r"\(", source, re.M)
    if not match:
        raise AssertionError(name)
    brace = source.index("{", match.start())
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end] + "\n"


PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
typedef uint64_t NvU64;
typedef unsigned NvU32;
#define UVM_SIZE_1MB (1024ULL * 1024)
#define UVM_PAGE_SIZE_64K 65536ULL
#define UVM_CHUNK_SIZE_MAX (2ULL * UVM_SIZE_1MB)
#define U64_MAX UINT64_MAX
#define UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED 1
#define READ_ONCE(x) (x)
#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))
#define min_t(t,a,b) ((t)(a) < (t)(b) ? (t)(a) : (t)(b))
#define clamp(v,lo,hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
#define UVM_ASSERT assert
typedef struct { volatile NvU64 numPages2m, numFreePages64k; } UvmPmaStatistics;
typedef struct { int state; } mock_chunk_t;
typedef struct {
    mock_chunk_t chunk;
    NvU64 access_heat, last_access_epoch, protected_until_ns, resident_since_ns;
    NvU32 migration_score;
} mock_root_t;
typedef struct {
    UvmPmaStatistics *pma_stats;
    int list_lock;
    struct { int work; NvU64 epoch; unsigned prefetch_count; bool active, recovering_watermark, stopping; } eager_eviction;
} uvm_pmm_gpu_t;
typedef struct { NvU64 used, high, target; } eager_eviction_usage_t;
static unsigned uvm_perf_eager_eviction_high_watermark = 90;
static unsigned uvm_perf_eager_eviction_low_watermark = 85;
static unsigned uvm_perf_eager_eviction_headroom_mb = 32;
static bool supported = true;
static bool eager_eviction_supported(uvm_pmm_gpu_t *p) { (void)p; return supported; }
static void eager_eviction_schedule(uvm_pmm_gpu_t *p, unsigned long d) { (void)p; (void)d; }
static void uvm_assert_spinlock_locked(int *p) { (void)p; }
static void uvm_spin_lock(int *p) { (void)p; }
static void uvm_spin_unlock(int *p) { (void)p; }
static void mod_delayed_work(void *w, void *d, unsigned long n) { (void)w; (void)d; (void)n; }
static void *system_wq;
'''


SCENARIOS = r'''
static UvmPmaStatistics stats;
static void setup(uvm_pmm_gpu_t *p, NvU64 roots, NvU64 free_pages) {
    memset(p, 0, sizeof(*p)); stats.numPages2m=roots; stats.numFreePages64k=free_pages; p->pma_stats=&stats;
}
int main(void) {
    uvm_pmm_gpu_t p;
    unsigned n;

    // 24 GiB: 393216 64K pages. Exact target is max(floor(85%), ceil(90%) - 512).
    setup(&p, 12288, 393216 - (353383 - 128));
    unsigned initial_grant = eager_eviction_prefetch_begin(&p, 4);
    assert(initial_grant == 4);
    assert(p.eager_eviction.prefetch_count == 1);
    assert(eager_eviction_prefetch_begin(&p, 4) == 0);
    eager_eviction_prefetch_end(&p);
    assert(p.eager_eviction.prefetch_count == 0);

    // Whole-root grants: 0, 1, 31, 32 and 33 pages of headroom.
    for (n=0; n<=33; ++n) {
        setup(&p, 100, 100*32 - (2720 - n));
        unsigned got = eager_eviction_prefetch_begin(&p, 99);
        assert(got == n / 32);
        if (got) eager_eviction_prefetch_end(&p);
    }

    // Exact PMA boundaries and pending-root arithmetic are conservative.
    setup(&p, 100, 100*32 - 2800);
    assert(eager_eviction_above_target(&p, 0));
    assert(!eager_eviction_above_target(&p, 3));
    setup(&p, 100, 0);
    assert(!eager_eviction_above_target(&p, UINT_MAX));

    // Reserve underflow and equal watermarks still leave one root below high.
    uvm_perf_eager_eviction_headroom_mb = UINT_MAX;
    setup(&p, 1, 0);
    assert(eager_eviction_prefetch_begin(&p, 4) == 0);
    uvm_perf_eager_eviction_headroom_mb = 32;
    uvm_perf_eager_eviction_low_watermark = 90;
    setup(&p, 100, 100*32 - 2848);
    assert(eager_eviction_prefetch_begin(&p, 4) == 0);
    uvm_perf_eager_eviction_low_watermark = 85;

    // Missing/zero PMA statistics preserve compatibility: admission is
    // serialized but does not silently disable speculative migration.
    setup(&p, 0, 0);
    p.pma_stats = NULL;
    unsigned fallback_grant = eager_eviction_prefetch_begin(&p, 4);
    assert(fallback_grant == 4);
    eager_eviction_prefetch_end(&p);
    supported = false;
    assert(eager_eviction_prefetch_begin(&p, 4) == 4);
    eager_eviction_prefetch_end(&p);
    puts("Passed: exact PMA headroom, whole-root grants, single-session admission, pending saturation, fallback and edge cases.");
    return 0;
}
'''


def main():
    source = SOURCE.read_text()
    code = PRELUDE
    code += extract(source, "eager_eviction_usage")
    code += extract(source, "eager_eviction_above_target")
    code += extract(source, "eager_eviction_prefetch_blocked_locked")
    # The production function is static in the source; use a renamed symbol in
    # the harness so its external mocked support predicate remains explicit.
    code += extract(source, "uvm_pmm_gpu_eager_eviction_prefetch_begin").replace(
        "uvm_pmm_gpu_eager_eviction_prefetch_begin", "eager_eviction_prefetch_begin")
    code += extract(source, "uvm_pmm_gpu_eager_eviction_prefetch_end").replace(
        "uvm_pmm_gpu_eager_eviction_prefetch_end", "eager_eviction_prefetch_end")
    code += SCENARIOS
    with tempfile.TemporaryDirectory(prefix="demand-paced-") as directory:
        path = Path(directory)
        (path / "test.c").write_text(code)
        subprocess.run(["cc", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", "-g", str(path / "test.c"),
                        "-o", str(path / "test")], check=True)
        subprocess.run([str(path / "test")], check=True)


if __name__ == "__main__":
    main()
