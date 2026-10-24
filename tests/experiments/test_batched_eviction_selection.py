import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).parents[2]
PMM = ROOT / "kernel-open/nvidia-uvm/uvm_pmm_gpu.c"


def extract_function(source, signature):
    start = source.index(signature)
    while source.find(";", start, source.find("{", start)) != -1:
        start = source.index(signature, start + len(signature))
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                assert source[end:end + 1] == "\n"
                return source[start:end + 1]
    raise AssertionError(signature)


def test_production_snapshot_with_asan_ubsan():
    production = PMM.read_text()
    names = [
        "allocated_chunk_precedes",
        "eager_snapshot_swap",
        "eager_snapshot_refill",
        "eager_snapshot_next",
        "eager_eviction_has_cold_used_candidate",
        "eager_eviction_one",
    ]
    extracted = {name: extract_function(production, "static " + ("bool " if name in ("allocated_chunk_precedes", "eager_eviction_has_cold_used_candidate", "eager_eviction_one") else "void " if name in ("eager_snapshot_swap", "eager_snapshot_refill") else "uvm_gpu_chunk_t *") + name) for name in names}
    typedef_start = production.index("typedef struct\n{\n    uvm_gpu_chunk_t *chunks[UVM_EAGER_EVICTION_SNAPSHOT_SIZE];")
    typedef_end = production.index("} eager_eviction_snapshot_t;", typedef_start) + len("} eager_eviction_snapshot_t;")
    typedef = production[typedef_start:typedef_end]

    source = r'''#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define UVM_EAGER_EVICTION_SNAPSHOT_SIZE 16
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define UVM_PMM_ALLOC_LIST_UNUSED 0
#define UVM_PMM_ALLOC_LIST_DISCARDED 1
#define UVM_PMM_ALLOC_LIST_USED 2
#define UVM_PMM_ALLOC_LIST_COUNT 3
#define U64_MAX UINT64_MAX
#define max(a,b) ((a) > (b) ? (a) : (b))
typedef uint64_t NvU64;
typedef struct chunk uvm_gpu_chunk_t;
typedef struct root uvm_gpu_root_chunk_t;
typedef struct pmm uvm_pmm_gpu_t;
struct list_head { uvm_gpu_chunk_t *first; };
struct chunk { struct root *root; struct chunk *next; int user, evictable; };
struct root {
    struct chunk chunk;
    NvU64 protected_until_ns, migration_score, resident_since_ns, access_heat, last_access_epoch;
};
struct roots { struct list_head alloc_list[UVM_PMM_ALLOC_LIST_COUNT]; NvU64 epoch; };
struct pmm { struct roots root_chunks; int list_lock, lock; struct list_head free_lists[2]; struct { NvU64 epoch; } eager_eviction; };
#define list_for_each_entry(pos, head, member) for ((pos) = (head)->first; (pos); (pos) = (pos)->next)
#define list_empty(head) ((head)->first == NULL)
static uvm_gpu_root_chunk_t *root_chunk_from_chunk(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *c) { (void)pmm; return c->root; }
static int uvm_gpu_chunk_is_user(uvm_gpu_chunk_t *c) { return c->user; }
static int chunk_is_evictable(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *c) { (void)pmm; return c->evictable; }
#define uvm_assert_spinlock_locked(lock) ((void)(lock))
static NvU64 NV_GETTIME(void) { return 20; }
static int uvm_perf_migration_aware_eviction_enable = 1;
static unsigned uvm_perf_eager_eviction_stale_intervals = 5;
static unsigned uvm_perf_eager_eviction_heat_ratio = 2;
typedef int NV_STATUS;
#define NV_OK 0
#define UVM_PMM_GPU_MEMORY_TYPE_USER 0
#define UVM_CHUNK_SIZE_MAX 2097152
#define UVM_PMM_LIST_NO_ZERO 0
#define UVM_PMM_LIST_ZERO 1
#define PMM_CONTEXT_DEFAULT 0
static unsigned claimed, evicted;
static void uvm_mutex_lock(int *p) { assert(!*p); *p=1; }
static void uvm_mutex_unlock(int *p) { assert(*p); *p=0; }
static void uvm_spin_lock(int *p) { assert(!*p); *p=1; }
static void uvm_spin_unlock(int *p) { assert(*p); *p=0; }
static struct list_head *find_free_list(uvm_pmm_gpu_t *p,int type,unsigned size,int zero) {
    (void)type; (void)size; return &p->free_lists[zero];
}
static uvm_gpu_chunk_t *list_first_chunk(struct list_head *p) { return p->first; }
static void remove_from(struct list_head *head,uvm_gpu_chunk_t *chunk) {
    uvm_gpu_chunk_t **link=&head->first;
    while(*link && *link!=chunk) link=&(*link)->next;
    if(*link) *link=chunk->next;
}
static void chunk_start_eviction(uvm_pmm_gpu_t *p,uvm_gpu_chunk_t *chunk) {
    assert(p->list_lock && p->lock && chunk->user && chunk->evictable);
    for(unsigned i=0;i<3;++i) remove_from(&p->root_chunks.alloc_list[i],chunk);
    for(unsigned i=0;i<2;++i) remove_from(&p->free_lists[i],chunk);
    chunk->next=NULL;chunk->evictable=0;++claimed;
}
static NV_STATUS evict_root_chunk(uvm_pmm_gpu_t *p,uvm_gpu_root_chunk_t *root,int context) {
    (void)root;(void)context;assert(p->lock && !p->list_lock);++evicted;return NV_OK;
}

'''
    source += typedef + "\n"
    source += "static bool eager_eviction_has_cold_used_candidate(uvm_pmm_gpu_t *pmm);\n"
    for name in names:
        source += extracted[name] + "\n"
    source += r'''
int main(void) {
    struct pmm p; struct root roots[97]; struct chunk *previous;
    struct chunk *selected[16], *eligible[97], *expected[16];
    uint32_t state = 11; size_t i, j, count, eligible_count;
    unsigned capacity, iteration;
    for (iteration = 0; iteration < 1000; ++iteration) {
        p = (struct pmm){0}; previous = NULL; eligible_count = 0;
        for (i = 0; i < 97; ++i) {
            state = state * 1664525u + 1013904223u;
            roots[i] = (struct root){0};
            roots[i].chunk.root = &roots[i]; roots[i].chunk.next = NULL;
            roots[i].chunk.user = (state & 31) != 0;
            roots[i].chunk.evictable = (state & 63) != 0;
            roots[i].migration_score = state % 17;
            roots[i].resident_since_ns = state % 101;
            roots[i].access_heat = state % 1001;
            roots[i].last_access_epoch = i;
            roots[i].protected_until_ns = (iteration % 3 == 0) ? 1000 :
                                          (iteration % 3 == 1) ? 0 : state % 201;
            if (previous) previous->next = &roots[i].chunk;
            else p.root_chunks.alloc_list[2].first = &roots[i].chunk;
            previous = &roots[i].chunk;
            eligible[eligible_count++] = &roots[i].chunk;
        }
        for (capacity = 1; capacity <= 16; capacity = capacity == 1 ? 8 : capacity == 8 ? 16 : 17) {
            eager_eviction_snapshot_t snapshot = { .capacity = capacity };
            eager_snapshot_refill(&p, &snapshot);
            assert(snapshot.count <= capacity);
            if (snapshot.count) {
                struct chunk *invalid = snapshot.chunks[0];
                invalid->evictable = 0;
                assert(eager_snapshot_next(&p, &snapshot) != invalid);
                invalid->evictable = 1;
            }
            eager_snapshot_refill(&p, &snapshot);
            for (i = 1; i < eligible_count; ++i) {
                struct chunk *value = eligible[i]; j = i;
                while (j && allocated_chunk_precedes(NULL, value, eligible[j - 1], 20)) {
                    eligible[j] = eligible[j - 1]; --j;
                }
                eligible[j] = value;
            }
            count = 0;
            while (count < capacity && (selected[count] = eager_snapshot_next(&p, &snapshot)))
                ++count;
            j = 0;
            for (i = 0; i < capacity && i < eligible_count; ++i) {
                if (eligible[i]->user && eligible[i]->evictable)
                    expected[j++] = eligible[i];
            }
            assert(count == j);
            for (i = 0; i < count; ++i) assert(selected[i] == expected[i]);
        }
    }
    // Real claim orchestration: consuming a priority root cannot turn a
    // uniformly hot used set into eligible non-forced replacement.
    p=(struct pmm){0};
    for(i=0;i<3;++i) {
        roots[i]=(struct root){0};roots[i].chunk.root=&roots[i];
        roots[i].chunk.user=roots[i].chunk.evictable=1;
        roots[i].access_heat=100;roots[i].last_access_epoch=0;
    }
    roots[0].chunk.next=&roots[1].chunk;
    p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_USED].first=&roots[0].chunk;
    p.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_UNUSED].first=&roots[2].chunk;
    eager_eviction_snapshot_t batch={.capacity=16};
    uvm_gpu_root_chunk_t *out=NULL;
    assert(eager_eviction_one(&p,false,&batch,&out) && out==&roots[2]);
    assert(claimed==1 && evicted==1 && !p.lock && !p.list_lock);
    assert(!eager_eviction_one(&p,false,&batch,&out));
    assert(claimed==1 && evicted==1 && !p.lock && !p.list_lock);
    assert(eager_eviction_one(&p,true,&batch,&out));
    assert(out==&roots[0] || out==&roots[1]);
    assert(claimed==2 && evicted==2 && !p.lock && !p.list_lock);
    return 0;
}'''
    with tempfile.TemporaryDirectory() as directory:
        c_file = pathlib.Path(directory) / "production.c"
        binary = pathlib.Path(directory) / "production"
        c_file.write_text(source)
        subprocess.run(["cc", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", str(c_file), "-o", str(binary)], check=True, cwd=ROOT)
        subprocess.run([str(binary)], check=True)

    print("Production top-K, stale eligibility and priority/coldness claim tests passed (1000 randomized sets).")


if __name__ == "__main__":
    test_production_snapshot_with_asan_ubsan()

