#!/usr/bin/env python3
"""Compile production queue/orchestrator code with a small UVM mock harness."""

import pathlib
import subprocess
import tempfile

SOURCE = pathlib.Path(__file__).parents[2] / "kernel-open/nvidia-uvm/uvm_gpu_access_counters.c"


def extract_function(text, name):
    start = text.index(name)
    brace = text.index("{", start)
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos + 1]
    raise ValueError(name)


PRELUDE = r'''
#include <assert.h>
#include <stddef.h>
#include <string.h>
typedef unsigned int NvU32;
typedef unsigned long long NvU64;
typedef int NV_STATUS;
#define NV_OK 0
#define UVM_ASSERT(x) assert(x)
struct mm_struct { int id; };
typedef struct uvm_gpu_struct { int id; int pmm; } uvm_gpu_t;
typedef struct uvm_va_space_struct { int id; } uvm_va_space_t;
typedef struct uvm_gpu_va_space_struct { uvm_va_space_t *va_space; uvm_gpu_t *gpu; } uvm_gpu_va_space_t;
typedef struct uvm_va_block_struct { int id; } uvm_va_block_t;
typedef struct uvm_parent_gpu_struct { int id; } uvm_parent_gpu_t;
typedef struct entry_struct { uvm_va_space_t *va_space; uvm_gpu_t *gpu; NvU64 address; int kind; int demand_status; int flags_status; uvm_va_block_t *block; } uvm_access_counter_buffer_entry_t;
typedef struct batch_struct { uvm_access_counter_buffer_entry_t *notifications[32]; NvU32 num_notifications; } uvm_access_counter_service_batch_context_t;
typedef uvm_access_counter_service_batch_context_t batch_t;
typedef struct access_struct { uvm_parent_gpu_t *parent_gpu; uvm_access_counter_service_batch_context_t batch_service_context; } uvm_access_counter_buffer_t;
static int log_events[256]; static size_t log_count; static int log_event(int kind, int id) { return log_events[log_count++] = kind * 1000 + id; }
static int current_kind;
static void uvm_cpu_block_policy_prefetch_on_signal(uvm_va_block_t *b, uvm_gpu_t *g, struct mm_struct *m) { (void)g; (void)m; log_event(3, b->id); }
static void preprocess_notifications(uvm_parent_gpu_t *p, batch_t *b) { (void)p; (void)b; }
static struct mm_struct global_mm;
static struct mm_struct *uvm_va_space_mm_retain_lock(uvm_va_space_t *v) { (void)v; return &global_mm; }
static void uvm_va_space_down_read(uvm_va_space_t *v) { (void)v; }
static void uvm_va_space_up_read(uvm_va_space_t *v) { (void)v; }
static void uvm_va_space_mm_release_unlock(uvm_va_space_t *v, struct mm_struct *m) { (void)v; (void)m; }
static uvm_gpu_va_space_t global_gpu_va_space;
static uvm_gpu_va_space_t *uvm_gpu_va_space_get(uvm_va_space_t *v, uvm_gpu_t *g) { global_gpu_va_space.va_space=v; global_gpu_va_space.gpu=g; return &global_gpu_va_space; }
static int uvm_va_space_has_access_counter_migrations(uvm_va_space_t *v) { (void)v; return 1; }
static int should_service_4k_notification(uvm_va_space_t *v, uvm_gpu_t *g, unsigned long long a) { (void)v; (void)g; (void)a; return 1; }
static NV_STATUS notify_tools_broadcast_and_process_flags(uvm_access_counter_buffer_t *a, uvm_access_counter_buffer_entry_t **e, int n, int f) { (void)a; (void)e; (void)n; (void)f; return log_event(4, 1), NV_OK; }
static NV_STATUS notify_tools_and_process_flags(uvm_va_space_t *v, uvm_gpu_t *g, uvm_access_counter_buffer_t *a, unsigned long long b, uvm_access_counter_buffer_entry_t **e, NvU32 n, NvU32 f, void *m) { (void)v; (void)g; (void)a; (void)b; (void)e; (void)n; (void)f; (void)m; return NV_OK; }
'''


def main():
    text = SOURCE.read_text()
    queue = text[text.index("#define UVM_DEMAND_FIRST_PREFETCH_QUEUE_CAPACITY"):text.index("// Each page in a tracked physical range")]
    orchestrator = extract_function(text, "static NV_STATUS service_notifications(uvm_access_counter_buffer_t")
    mock_batch = r'''
static NV_STATUS service_notifications_batch(uvm_gpu_va_space_t *g, struct mm_struct *mm, uvm_access_counter_buffer_t *a, NvU32 index, NvU32 *out, demand_first_prefetch_queue_t *q) {
    uvm_access_counter_buffer_entry_t *e = a->batch_service_context.notifications[index];
    (void)g; (void)mm; current_kind = e->kind; *out = index + 1;
    if (e->kind == 1) { log_event(5, e->block->id); return NV_OK; } /* ATS */
    log_event(1, e->block->id); /* direct demand */
    if (e->demand_status != NV_OK) return e->demand_status;
    if (e->flags_status != NV_OK) return e->flags_status;
    log_event(2, e->block->id); /* notification flags */
    demand_first_prefetch_queue_push(q, e->block, e->gpu, mm);
    return NV_OK;
}
'''
    tests = r'''
static void reset(void) { memset(log_events, 0, sizeof(log_events)); log_count = 0; }
static void run(uvm_access_counter_buffer_t *a, uvm_access_counter_buffer_entry_t **e, size_t n) { a->batch_service_context.num_notifications = (NvU32)n; for (size_t i=0;i<n;i++) a->batch_service_context.notifications[i]=e[i]; }
int main(void) {
    uvm_parent_gpu_t parent={0}; uvm_va_space_t va1={1}, va2={2}; uvm_gpu_t gpu1={1,0}, gpu2={2,0}; struct { uvm_va_block_t b; uvm_access_counter_buffer_entry_t e; } items[10]; uvm_access_counter_buffer_t a={.parent_gpu=&parent}; uvm_access_counter_buffer_entry_t *entries[10];
    for (int i=0;i<10;i++) { items[i].b.id=i+1; items[i].e=(uvm_access_counter_buffer_entry_t){&va1,&gpu1,0,0,NV_OK,NV_OK,&items[i].b}; entries[i]=&items[i].e; }
    reset(); run(&a,entries,2); items[0].e.va_space=&va1; items[1].e.va_space=&va1; assert(service_notifications(&a)==NV_OK); assert(log_events[0]==1001 && log_events[1]==2001 && log_events[2]==1002 && log_events[3]==2002 && log_count==6 && log_events[4]==3001 && log_events[5]==3002);
    reset(); items[2].e.gpu=&gpu2; items[3].e.gpu=&gpu2; items[2].e.va_space=&va1; items[3].e.va_space=&va2; run(&a,(uvm_access_counter_buffer_entry_t *[]){&items[2].e,&items[3].e},2); assert(service_notifications(&a)==NV_OK); assert(log_events[2]==3003 && log_events[5]==3004);
    reset(); items[0].e.demand_status=7; items[1].e.demand_status=NV_OK; run(&a,(uvm_access_counter_buffer_entry_t *[]){&items[0].e,&items[1].e},2); assert(service_notifications(&a)==7 && log_count==1);
    reset(); items[0].e.demand_status=NV_OK; items[0].e.flags_status=8; run(&a,(uvm_access_counter_buffer_entry_t *[]){&items[0].e},1); assert(service_notifications(&a)==8 && log_count==1);
    reset(); for (int i=0;i<9;i++) { items[i].e.kind=0; items[i].e.demand_status=NV_OK; items[i].e.flags_status=NV_OK; items[i].e.va_space=&va1; items[i].e.gpu=&gpu1; entries[i]=&items[i].e; } run(&a,entries,9); assert(service_notifications(&a)==NV_OK && log_count==27 && log_events[18]==3001 && log_events[26]==3009);
    reset(); items[0].e.kind=1; items[0].e.va_space=&va1; items[1].e.va_space=NULL; run(&a,(uvm_access_counter_buffer_entry_t *[]){&items[0].e,&items[1].e},2); assert(service_notifications(&a)==NV_OK && log_count==2 && log_events[0]==5001 && log_events[1]==4001);
    return 0;
}
'''
    source_text = PRELUDE + queue + mock_batch + orchestrator + tests
    with tempfile.TemporaryDirectory(prefix="demand-first-test-") as directory:
        root = pathlib.Path(directory); cfile = root / "harness.c"; binary = root / "harness"
        cfile.write_text(source_text)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", "-g", str(cfile), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

    block_prelude = r'''
#include <assert.h>
#include <stddef.h>
typedef unsigned int NvU32; typedef unsigned long long NvU64; typedef int NV_STATUS;
#define NV_OK 0
#define UVM_ACCESS_COUNTER_ACTION_BATCH_CLEAR 1
#define UVM_ASSERT(x) assert(x)
struct mm_struct { int id; }; typedef int uvm_page_mask_t; typedef int uvm_gpu_chunk_t;
typedef struct uvm_gpu_struct { int id; int pmm; } uvm_gpu_t;
typedef struct uvm_va_space_struct { int id; int lock; } uvm_va_space_t;
typedef struct uvm_gpu_va_space_struct { uvm_va_space_t *va_space; uvm_gpu_t *gpu; } uvm_gpu_va_space_t;
typedef struct uvm_va_block_struct { NvU64 end; int id; int lock; } uvm_va_block_t;
typedef struct uvm_va_block_context_struct { int id; } uvm_va_block_context_t;
typedef struct uvm_service_block_context_struct { uvm_va_block_context_t *block_context; int access_counters_buffer_index; } uvm_service_block_context_t;
typedef struct entry_struct { uvm_va_space_t *va_space; uvm_gpu_t *gpu; NvU64 address; int counter_value; } uvm_access_counter_buffer_entry_t;
typedef struct batch_struct { uvm_access_counter_buffer_entry_t *notifications[4]; NvU32 num_notifications; uvm_page_mask_t accessed_pages; uvm_service_block_context_t block_service_context; } uvm_access_counter_service_batch_context_t;
typedef struct access_struct { int index; uvm_access_counter_service_batch_context_t batch_service_context; } uvm_access_counter_buffer_t;
static int demand_status, flags_status, queue_count, prefetch_ids[4];
static void uvm_cpu_block_policy_prefetch_on_signal(uvm_va_block_t *b, uvm_gpu_t *g, struct mm_struct *m) { (void)g; (void)m; prefetch_ids[queue_count++] = b->id; }
static void uvm_page_mask_zero(uvm_page_mask_t *m) { *m = 0; }
static void uvm_assert_rwsem_locked(void *l) { (void)l; }
static void uvm_va_block_context_init(uvm_va_block_context_t *c, struct mm_struct *m) { (void)c; (void)m; }
static void uvm_mutex_lock(void *m) { (void)m; } static void uvm_mutex_unlock(void *m) { (void)m; }
static void expand_notification_block(uvm_gpu_va_space_t *g, uvm_va_block_t *b, uvm_access_counter_buffer_t *a, uvm_va_block_context_t *c, uvm_page_mask_t *p, uvm_access_counter_buffer_entry_t *e) { (void)g;(void)b;(void)a;(void)c;(void)p;(void)e; }
static NV_STATUS service_notification_va_block_helper(struct mm_struct *m, uvm_va_block_t *b, uvm_gpu_t *g, uvm_access_counter_service_batch_context_t *c) { (void)m;(void)b;(void)g;(void)c; return demand_status; }
static uvm_gpu_chunk_t *uvm_va_block_lookup_gpu_chunk(uvm_va_block_t *b, uvm_gpu_t *g, NvU64 a) { (void)b;(void)g;(void)a; static uvm_gpu_chunk_t chunk; return &chunk; }
static void uvm_pmm_gpu_mark_chunk_accessed(void *p, uvm_gpu_chunk_t *c, int v) { (void)p;(void)c;(void)v; }
static NV_STATUS notify_tools_and_process_flags(uvm_va_space_t *v, uvm_gpu_t *g, uvm_access_counter_buffer_t *a, NvU64 x, uvm_access_counter_buffer_entry_t **e, NvU32 n, NvU32 f, void *m) { (void)v;(void)g;(void)a;(void)x;(void)e;(void)n;(void)f;(void)m; return flags_status; }
'''
    block_queue = text[text.index("#define UVM_DEMAND_FIRST_PREFETCH_QUEUE_CAPACITY"):text.index("// Each page in a tracked physical range")]
    block_function = extract_function(text, "static NV_STATUS service_notifications_in_block(uvm_gpu_va_space_t")
    block_tests = r'''
int main(void) {
    uvm_va_space_t va={1,0}; uvm_gpu_t gpu={2,0}; uvm_gpu_va_space_t gva={&va,&gpu}; uvm_va_block_t block={.end=99,.id=7}; uvm_va_block_context_t context={0}; struct mm_struct mm={0}; demand_first_prefetch_queue_t queue={0};
    uvm_access_counter_buffer_entry_t entry={&va,&gpu,10,3}; uvm_access_counter_buffer_t access={0}; access.index=4; access.batch_service_context.notifications[0]=&entry; access.batch_service_context.num_notifications=1; access.batch_service_context.block_service_context.block_context=&context;
    NvU32 out=0;
    demand_status=9; flags_status=NV_OK; queue_count=0; assert(service_notifications_in_block(&gva,&mm,&access,&block,0,&out,&queue)==9); assert(queue.count==0 && queue_count==0); demand_first_prefetch_queue_discard(&queue);
    demand_status=NV_OK; flags_status=8; queue_count=0; assert(service_notifications_in_block(&gva,&mm,&access,&block,0,&out,&queue)==8); assert(queue.count==0 && queue_count==0); demand_first_prefetch_queue_discard(&queue);
    demand_status=NV_OK; flags_status=NV_OK; queue_count=0; assert(service_notifications_in_block(&gva,&mm,&access,&block,0,&out,&queue)==NV_OK); assert(queue.count==1 && queue.blocks[0]==&block);
    demand_first_prefetch_queue_flush(&queue); assert(queue_count==1 && prefetch_ids[0]==7);
    return 0;
}
'''
    block_source = block_prelude + block_queue + block_function + block_tests
    with tempfile.TemporaryDirectory(prefix="demand-first-block-test-") as directory:
        root = pathlib.Path(directory); cfile = root / "block_harness.c"; binary = root / "block_harness"
        cfile.write_text(block_source)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", "-g", str(cfile), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
