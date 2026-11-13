#!/usr/bin/env python3
"""Execute production 4MiB migration and the real allocation-retry macro."""
from pathlib import Path
import subprocess
import tempfile
from test_demand_first_eviction import function

ROOT = Path(__file__).parents[2]
PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef uint64_t NvU64; typedef uint32_t NvU32; typedef int NV_STATUS;
#define NV_OK 0
#define NV_ERR_NO_MEMORY 2
#define NV_ERR_MORE_PROCESSING_REQUIRED 3
#define PAGE_SIZE 4096ULL
#define UVM_VA_BLOCK_SIZE (2ULL*1024*1024)
#define UVM_CPU_REGION_SIZE (2ULL*UVM_VA_BLOCK_SIZE)
#define UVM_MAKE_RESIDENT_CAUSE_ACCESS_COUNTER 7
#define UVM_PROT_READ_WRITE_ATOMIC 4
#define UvmEventMapRemoteCauseInvalid 0
#define NUMA_NO_NODE (-1)
#define UVM_ID_CPU 0
#define UVM_ID_IS_CPU(x) ((x)==0)
#define UVM_ASSERT assert
#ifdef __block
#undef __block
#endif
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
typedef struct { NvU64 bits[8]; } uvm_page_mask_t;
typedef struct { int unused; } uvm_tracker_t;
#define UVM_TRACKER_INIT() {0}
typedef struct { int unused; } uvm_va_block_retry_t;
typedef struct { struct { uvm_page_mask_t pages_changed_residency; unsigned all_involved_processors; unsigned access_counters_buffer_index; } make_resident; } uvm_va_block_context_t;
struct mm_struct { int unused; };
typedef struct { bool access_counters_supported,is_integrated_gpu; } parent_t;
typedef struct { parent_t *parent; int id; } uvm_gpu_t;
typedef struct { int lock; } space_t;
typedef struct { struct { struct { NvU64 start,end; } node; space_t *va_space; } va_range; bool cpu_access_counter_policy; struct { int preferred_location; } policy; } uvm_va_range_managed_t;
typedef struct { int lock; unsigned mapped; NvU64 start,end; uvm_va_range_managed_t *managed_range; uvm_tracker_t tracker; uvm_page_mask_t cpu,gpu,thrash; bool populated; } uvm_va_block_t;
static uvm_va_block_t store[4],*blocks[4];
static uvm_va_range_managed_t range;
static parent_t parent;
static uvm_gpu_t gpu;
static space_t space;
static bool alloc_failure,map_retry,make_retry,unmap_during_alloc,unmap_on_retry,unmap_after_trigger;
static NV_STATUS make_error,map_error,dep_error,wait_error;
static unsigned held,make_calls,map_calls,wait_calls,dep_calls,frees,retry_deinits;
static NvU64 visited[16];
static unsigned request_pages[16],mapped_pages[16];
static NvU64 uvm_cpu_preferred_region_failures,uvm_cpu_preferred_region_migrated_bytes,uvm_cpu_preferred_region_promotions;
#define atomic64_inc(p) (++*(p))
#define atomic64_add(n,p) (*(p)+=(n))
static bool uvm_va_block_is_hmm(uvm_va_block_t *b) {(void)b;return false;}
static bool gpu_supported(space_t *s,uvm_gpu_t *g) {return s && g && g->parent->access_counters_supported && !g->parent->is_integrated_gpu;}
static void uvm_assert_rwsem_locked_read(int *l) {assert(*l);}
static void uvm_assert_mutex_locked(int *l) {assert(*l && held==1);}
static void uvm_mutex_lock(int *l) {assert(!*l && !held);*l=1;held=1;}
static void uvm_mutex_unlock(int *l) {assert(*l && held==1);*l=0;held=0;}
static size_t uvm_va_range_block_index(uvm_va_range_managed_t *r,NvU64 a)
{assert(a>=r->va_range.node.start && a<=r->va_range.node.end);return (a-(r->va_range.node.start & ~(UVM_VA_BLOCK_SIZE-1)))/UVM_VA_BLOCK_SIZE;}
static uvm_va_block_t *uvm_va_range_block(uvm_va_range_managed_t *r,size_t i) {(void)r;assert(i<4);return blocks[i];}
static void uvm_page_mask_zero(uvm_page_mask_t *m) {memset(m,0,sizeof(*m));}
static void uvm_page_mask_copy(uvm_page_mask_t *d,const uvm_page_mask_t *s) {*d=*s;}
static void uvm_page_mask_andnot(uvm_page_mask_t *d,const uvm_page_mask_t *a,const uvm_page_mask_t *b) {for(unsigned i=0;i<8;++i)d->bits[i]=a->bits[i]&~b->bits[i];}
static unsigned uvm_page_mask_weight(const uvm_page_mask_t *m) {unsigned n=0;for(unsigned i=0;i<8;++i)n+=__builtin_popcountll(m->bits[i]);return n;}
static bool uvm_page_mask_empty(const uvm_page_mask_t *m) {return !uvm_page_mask_weight(m);}
static const uvm_page_mask_t *uvm_va_block_resident_mask_get(uvm_va_block_t *b,int id,int nid) {assert(b->lock && nid==-1);return id==0?(b->populated?&b->cpu:NULL):&b->gpu;}
static const uvm_page_mask_t *uvm_perf_thrashing_get_thrashing_pages(uvm_va_block_t *b) {assert(b->lock);return &b->thrash;}
static bool uvm_processor_mask_test(unsigned *m,int id) {assert(held==1);return (*m & (1U<<id))!=0;}
static void uvm_processor_mask_zero(unsigned *m) {*m=0;}
static void uvm_va_block_context_init(uvm_va_block_context_t *c,struct mm_struct *m) {(void)m;memset(c,0,sizeof(*c));}
static uvm_va_block_context_t context;
static uvm_va_block_context_t *uvm_va_block_context_alloc(struct mm_struct *m) {(void)m;if(unmap_during_alloc)for(unsigned i=0;i<4;++i)store[i].mapped=0;return alloc_failure?NULL:&context;}
static void uvm_va_block_context_free(uvm_va_block_context_t *c) {assert(c==&context && !held && wait_calls);++frees;}
static void uvm_va_block_retry_init(uvm_va_block_retry_t *r) {memset(r,0,sizeof(*r));}
static void uvm_va_block_retry_deinit(uvm_va_block_retry_t *r,uvm_va_block_t *b) {assert(r && b->lock);++retry_deinits;}
static int uvm_va_block_region_from_block(uvm_va_block_t *b) {assert(b->lock);return 0;}
static NV_STATUS uvm_va_block_make_resident(uvm_va_block_t *b,uvm_va_block_retry_t *r,uvm_va_block_context_t *c,
 int id,int region,const uvm_page_mask_t *pages,void *unused,int cause)
{
 (void)r; assert(b->lock && held==1 && id==1 && !region && !unused && cause==7);
 assert(c->make_resident.access_counters_buffer_index==5 && make_calls<16);
 visited[make_calls]=b->start;request_pages[make_calls]=uvm_page_mask_weight(pages);++make_calls;
 b->mapped=0; // make_resident removes the old remote GPU mappings before population.
 if(unmap_on_retry) {unmap_on_retry=false;return NV_ERR_MORE_PROCESSING_REQUIRED;}
 if(make_error)return make_error;
 uvm_page_mask_andnot(&c->make_resident.pages_changed_residency,pages,&b->gpu);
 if(make_retry) { // A partial migration followed by allocation retry.
  make_retry=false;for(unsigned i=1;i<8;++i)c->make_resident.pages_changed_residency.bits[i]=0;
  c->make_resident.pages_changed_residency.bits[0]&=1;
 }
 for(unsigned i=0;i<8;++i) {
  b->gpu.bits[i]|=c->make_resident.pages_changed_residency.bits[i];
  b->cpu.bits[i]&=~c->make_resident.pages_changed_residency.bits[i];
 }
 return uvm_page_mask_weight(&b->gpu)<uvm_page_mask_weight(pages)?NV_ERR_MORE_PROCESSING_REQUIRED:NV_OK;
}
static NV_STATUS uvm_va_block_map(uvm_va_block_t *b,uvm_va_block_context_t *c,int id,int region,
 const uvm_page_mask_t *pages,int prot,int cause,uvm_tracker_t *t)
{
 (void)c;assert(b->lock && held==1 && id==1 && !region && prot==4 && !cause && t==&b->tracker && map_calls<16);
 mapped_pages[map_calls++]=uvm_page_mask_weight(pages);
 if(map_retry) {map_retry=false;return NV_ERR_MORE_PROCESSING_REQUIRED;}
 if(!map_error)b->mapped=2;
 if(unmap_after_trigger) {unmap_after_trigger=false;b->mapped=0;}
 return map_error;
}
static NV_STATUS uvm_tracker_add_tracker_safe(uvm_tracker_t *d,uvm_tracker_t *s) {assert(d && s && held==1);++dep_calls;return dep_error;}
static NV_STATUS uvm_tracker_wait_deinit(uvm_tracker_t *t) {assert(t && !held);++wait_calls;return wait_error;}
'''
CASES = r'''
static void reset(NvU64 start,NvU64 end)
{
 memset(store,0,sizeof(store));memset(blocks,0,sizeof(blocks));memset(&range,0,sizeof(range));
 parent=(parent_t){true,false};gpu=(uvm_gpu_t){&parent,1};space.lock=1;
 range.va_range.va_space=&space;range.va_range.node.start=start;range.va_range.node.end=end;
 range.cpu_access_counter_policy=true;range.policy.preferred_location=0;
 NvU64 aligned=start & ~(UVM_VA_BLOCK_SIZE-1);
 for(unsigned i=0;i<4 && aligned+i*UVM_VA_BLOCK_SIZE<=end;++i) {
  uvm_va_block_t *b=&store[i];blocks[i]=b;b->managed_range=&range;b->populated=true;b->mapped=2;
  b->start=max(start,aligned+i*UVM_VA_BLOCK_SIZE);b->end=min(end,aligned+(i+1)*UVM_VA_BLOCK_SIZE-1);
  unsigned pages=(b->end-b->start+1)/PAGE_SIZE;
  for(unsigned j=0;j<pages;++j)b->cpu.bits[j/64]|=1ULL<<(j%64);
 }
 alloc_failure=map_retry=make_retry=unmap_during_alloc=unmap_on_retry=unmap_after_trigger=false;make_error=map_error=dep_error=wait_error=0;
 held=make_calls=map_calls=wait_calls=dep_calls=frees=retry_deinits=0;
 memset(visited,0,sizeof(visited));memset(request_pages,0,sizeof(request_pages));memset(mapped_pages,0,sizeof(mapped_pages));
 uvm_cpu_preferred_region_failures=uvm_cpu_preferred_region_migrated_bytes=uvm_cpu_preferred_region_promotions=0;
}
static NV_STATUS run(unsigned trigger,bool *did) {return uvm_cpu_block_policy_migrate_region_on_signal(&store[trigger],&gpu,NULL,5,did);}
int main(void)
{
 bool did;
 for(unsigned trigger=0;trigger<4;++trigger) {
  reset(0,4*UVM_VA_BLOCK_SIZE-1);assert(run(trigger,&did)==NV_OK && did);
  assert(make_calls==2 && map_calls==2 && wait_calls==1 && frees==1 && retry_deinits==2);
  assert(visited[0]==trigger*UVM_VA_BLOCK_SIZE && visited[1]==(trigger^1)*UVM_VA_BLOCK_SIZE);
  assert(uvm_cpu_preferred_region_migrated_bytes==2*UVM_VA_BLOCK_SIZE && uvm_cpu_preferred_region_promotions==1);
  assert(run(trigger,&did)==NV_OK && !did && make_calls==2 && map_calls==2); // Idempotent.
 }
 reset(UVM_VA_BLOCK_SIZE+PAGE_SIZE,3*UVM_VA_BLOCK_SIZE-1);
 assert(run(0,&did)==0 && did && make_calls==1 && visited[0]==UVM_VA_BLOCK_SIZE+PAGE_SIZE);
 assert(uvm_cpu_preferred_region_migrated_bytes==UVM_VA_BLOCK_SIZE-PAGE_SIZE);
 reset(0,UVM_VA_BLOCK_SIZE+PAGE_SIZE-1);assert(run(1,&did)==0 && did && make_calls==2);
 assert(uvm_cpu_preferred_region_migrated_bytes==UVM_VA_BLOCK_SIZE+PAGE_SIZE);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);blocks[0]=NULL;assert(run(1,&did)==0 && did && make_calls==1);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);store[1].populated=false;assert(run(0,&did)==0 && make_calls==1);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);store[0].thrash=store[0].cpu;store[1].thrash=store[1].cpu;
 assert(run(0,&did)==0 && !did && !make_calls);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);uvm_page_mask_zero(&store[0].cpu);store[0].cpu.bits[0]=7;
 store[0].gpu.bits[0]=1;store[0].thrash.bits[0]=4;uvm_page_mask_zero(&store[1].cpu);
 assert(run(0,&did)==0 && did && make_calls==1 && request_pages[0]==1);
 assert(uvm_cpu_preferred_region_migrated_bytes==PAGE_SIZE);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);map_retry=true;assert(run(0,&did)==0 && did && make_calls==3 && map_calls==3);
 assert(request_pages[0]==512 && request_pages[1]==512 && mapped_pages[0]==512 && mapped_pages[1]==512);
 assert(uvm_cpu_preferred_region_migrated_bytes==2*UVM_VA_BLOCK_SIZE);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);make_retry=true;assert(run(0,&did)==0 && did && make_calls==3 && map_calls==2);
 assert(request_pages[0]==512 && request_pages[1]==512 && uvm_cpu_preferred_region_migrated_bytes==2*UVM_VA_BLOCK_SIZE);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);alloc_failure=true;assert(run(0,&did)==NV_ERR_NO_MEMORY && !did && !wait_calls && !make_calls && !frees);
 assert(uvm_cpu_preferred_region_failures==1);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);make_error=7;assert(run(0,&did)==7 && !did && make_calls==1 && dep_calls==1 && wait_calls==1 && frees==1);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);map_error=8;assert(run(0,&did)==8 && did && make_calls==1 && wait_calls==1 && frees==1);
 assert(uvm_cpu_preferred_region_failures==1 && !uvm_cpu_preferred_region_promotions);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);dep_error=9;assert(run(0,&did)==9 && did && make_calls==1 && wait_calls==1 && frees==1);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);wait_error=10;assert(run(0,&did)==10 && did && make_calls==2 && wait_calls==1 && frees==1);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);store[0].mapped=0;
 assert(run(0,&did)==0 && !did && !make_calls && !wait_calls && !frees);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);unmap_during_alloc=true;
 assert(run(0,&did)==0 && !did && !make_calls && wait_calls==1 && frees==1);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);unmap_on_retry=true;
 assert(run(0,&did)==0 && did && make_calls==3 && map_calls==2 && wait_calls==1 && frees==1);
 assert(store[0].mapped==2 && store[1].mapped==2);
 reset(0,4*UVM_VA_BLOCK_SIZE-1);unmap_after_trigger=true;
 assert(run(0,&did)==0 && did && make_calls==1 && map_calls==1 && wait_calls==1);
 for(unsigned which=0;which<3;++which) {
  reset(0,4*UVM_VA_BLOCK_SIZE-1);
  if(which==0)parent.access_counters_supported=false;
  if(which==1)parent.is_integrated_gpu=true;
  if(which==2)range.policy.preferred_location=1;
  assert(run(0,&did)==0 && !did && !make_calls && !wait_calls);
 }
 puts("4MiB regions: exact populated bytes, VA alignment/boundaries, single locks, retries, mapping, waits and failures passed.");
 return 0;
}
'''

def main():
    source = (ROOT / 'kernel-open/nvidia-uvm/uvm_cpu_block_policy.c').read_text()
    header = (ROOT / 'kernel-open/nvidia-uvm/uvm_va_block.h').read_text()
    start = header.index('#define UVM_VA_BLOCK_RETRY_LOCKED(')
    retry = header[start:header.index('\n})', start) + 3]
    names = ('should_handle_gpu_signal', 'uvm_cpu_block_policy_should_migrate_region',
             'migrate_region_block_locked', 'uvm_cpu_block_policy_migrate_region_on_signal')
    code = PRELUDE + retry + '\n' + ''.join(function(source, name) for name in names) + CASES
    with tempfile.TemporaryDirectory(prefix='region-migration-') as directory:
        c=Path(directory)/'test.c'; binary=Path(directory)/'test'; c.write_text(code)
        subprocess.run(['cc','-std=gnu11','-O1','-Wall','-Wextra','-Werror',
                        '-fsanitize=address,undefined','-g',str(c),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)

if __name__ == '__main__':
    main()
