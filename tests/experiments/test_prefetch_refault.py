#!/usr/bin/env python3
"""Exercise production refault-distance admission, including progress and isolation."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HARNESS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stddef.h>
#include "uvm_prefetch_refault.h"
#define M (1ULL<<20)
#define B (2*M)
static const uvm_prefetch_refault_config_t cfg={.capacity_bytes=64*M,.short_distance_percent=25,.max_neighbors=31};
static uvm_prefetch_refault_block_t states[32];
static void evict(uvm_prefetch_refault_t *t,unsigned block) {
    uvm_prefetch_refault_evict(t,&states[block]);
}
static void progress(uvm_prefetch_refault_t *t,NvU64 bytes) {
    uvm_prefetch_refault_demand(t,&cfg,NULL,99,0,bytes,0,0,false,false);
}
static void obs(uvm_prefetch_refault_t *t,NvU64 block,NvU64 batch) {
    uvm_prefetch_refault_demand(t,&cfg,&states[block],1,block*B,0,0,batch,true,true);
}
static void returns(void) {
    uvm_prefetch_refault_t t={0};
    memset(states,0,sizeof(states));
    assert(sizeof(t)<64*1024);
    evict(&t,8);
    progress(&t,3*M);
    evict(&t,8); // Must not refresh first partial eviction.
    progress(&t,2*M);
    uvm_prefetch_refault_demand(&t,&cfg,&states[8],1,8*B,4*M,0,0,true,true);
    assert(t.refaults==1 && t.last_distance==5*M && t.demand_bytes==9*M);
    assert(t.short_refaults==1);
    // Duplicate evidence still accounts actual new residency, never a second return.
    uvm_prefetch_refault_demand(&t,&cfg,&states[8],1,8*B,M,0,0,true,true);
    assert(t.refaults==1 && t.demand_bytes==10*M);
    evict(&t,8);
    progress(&t,M);
    obs(&t,8,1);
    assert(t.last_distance==M && t.short_refaults==2);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,8*B)==0);
    // Local suppression reopens after a quarter-capacity of demand traffic.
    progress(&t,16*M);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,8*B)==1);
    evict(&t,8);
    progress(&t,17*M);
    obs(&t,8,2);
    assert(t.last_distance==17*M && t.refaults==3 && t.short_refaults==2);
    assert(states[8].short_returns==0);
    // Failed native service advances progress but does not consume shadow/history.
    evict(&t,8);
    uvm_prefetch_refault_demand(&t,&cfg,&states[8],1,8*B,B,0,3,false,true);
    assert(t.refaults==3 && states[8].evicted);
    obs(&t,8,3);
    assert(t.refaults==4 && t.last_distance==B);
}
static void spatial(void) {
    uvm_prefetch_refault_t t={0};
    memset(states,0,sizeof(states));
    // Four distinct adjacent observations grow1->2. Repeated offset/batches cannot.
    for(unsigned i=0;i<5;i++) obs(&t,i,i);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==2);
    for(unsigned i=10;i<100;i++) obs(&t,4,i);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==2);
    for(unsigned i=5;i<32;i++) obs(&t,i,100+i);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==31);
    memset(&t,0,sizeof(t));memset(states,0,sizeof(states));
    for(unsigned i=0;i<10;i+=2) {
        obs(&t,i,i);
        uvm_prefetch_refault_population(&t,&cfg,1,(i+1)*B);
    }
    assert(t.growths==1 && uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==2);
    unsigned credits=uvm_prefetch_refault_tile(&t,1,0)->credits;
    for(unsigned i=10;i<32;i++) uvm_prefetch_refault_population(&t,&cfg,1,i*B);
    assert(uvm_prefetch_refault_tile(&t,1,0)->credits==credits && t.growths==1);
    // Notification direction is not treated as stream direction.
    memset(&t,0,sizeof(t));memset(states,0,sizeof(states));
    for(unsigned i=0;i<5;i++) obs(&t,31-i,i);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,31*B)==2);
    // Dispersed non-adjacent accesses do not earn growth.
    memset(&t,0,sizeof(t));memset(states,0,sizeof(states));
    for(unsigned i=0;i<16;i++) obs(&t,2*i,i);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==1);
    // Credits are per distinct offset per progress horizon, even across batches.
    obs(&t,1,50); obs(&t,3,51);
    assert(uvm_prefetch_refault_tile(&t,1,0)->credits==2);
    obs(&t,1,52);
    assert(uvm_prefetch_refault_tile(&t,1,0)->credits==2);
    progress(&t,16*M);
    (void)uvm_prefetch_refault_neighbors(&t,&cfg,1,0);
    assert(!uvm_prefetch_refault_tile(&t,1,0)->seen);
    assert(!uvm_prefetch_refault_tile(&t,1,0)->credits);
    // Inactive tiles do not retain a large prefetch window forever.
    uvm_prefetch_refault_tile(&t,1,0)->window=31;
    progress(&t,64*M);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==1);
}
static void pressure(void) {
    uvm_prefetch_refault_t t={0};
    memset(states,0,sizeof(states));
    // A single ping-pong block throttles its own tile, never another allocation.
    for(unsigned i=0;i<8;i++) {evict(&t,0); obs(&t,0,i);}
    assert(t.refaults==8 && t.short_refaults==8);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==0);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,2,64*M)==1);
    progress(&t,16*M);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==1);
    // Distinct short-returning blocks suppress only their shared tile.
    memset(&t,0,sizeof(t));memset(states,0,sizeof(states));
    for(unsigned i=0;i<3;i++) {evict(&t,i);obs(&t,i,i);}
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==1);
    evict(&t,3);obs(&t,3,3);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==0);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,64*M)==1);
    // Expiry uses elapsed bytes; crossing an absolute epoch boundary is insufficient.
    t.demand_bytes=15*M;
    uvm_prefetch_refault_tile(&t,1,0)->epoch_start=t.demand_bytes;
    progress(&t,2*M);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==0);
    progress(&t,14*M);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==1);
    // Two shorts separated by a VRAM worth of demand progress are not consecutive.
    memset(&t,0,sizeof(t));memset(states,0,sizeof(states));
    evict(&t,0);obs(&t,0,0);progress(&t,64*M);evict(&t,0);obs(&t,0,1);
    assert(states[0].short_returns==1);
    assert(uvm_prefetch_refault_neighbors(&t,&cfg,1,0)==1);
}
static void bounds_isolation(void) {
    uvm_prefetch_refault_t t={0};
    memset(states,0,sizeof(states));
    uvm_prefetch_refault_config_t c=cfg;
    assert(uvm_prefetch_refault_short(&c,16*M));
    assert(!uvm_prefetch_refault_short(&c,16*M+1));
    c.capacity_bytes=128*M;
    assert(uvm_prefetch_refault_short(&c,32*M));
    assert(!uvm_prefetch_refault_short(&c,32*M+1));
    c.capacity_bytes=103;
    assert(uvm_prefetch_refault_short(&c,25) && !uvm_prefetch_refault_short(&c,26));
    c.capacity_bytes=UINT64_MAX;c.short_distance_percent=100;
    assert(uvm_prefetch_refault_short(&c,UINT64_MAX));
    c.capacity_bytes=0;
    assert(!uvm_prefetch_refault_short(&c,0));
    assert(uvm_prefetch_refault_neighbors(&t,&c,1,0)==0);
    // State belongs to the actual virtual block/GPU, avoiding shadow collisions.
    uvm_prefetch_refault_block_t other={0};
    evict(&t,0);
    uvm_prefetch_refault_demand(&t,&cfg,&other,2,0,0,0,0,true,true);
    assert(!t.refaults);
    // An unrelated CPU page in the same partially evicted block is not a return.
    uvm_prefetch_refault_demand(&t,&cfg,&states[0],1,0,0,0,0,true,false);
    assert(!t.refaults && states[0].evicted);
    obs(&t,0,1);assert(t.refaults==1);
    // Zero-initialized split GPU state does not inherit lower-half timestamps.
    assert(!other.evicted && !other.short_returns);
    // Byte-clock wrap keeps elapsed distances valid modulo2^64.
    memset(&t,0,sizeof(t));memset(states,0,sizeof(states));t.demand_bytes=UINT64_MAX-M;
    evict(&t,0);progress(&t,2*M);obs(&t,0,0);
    assert(t.last_distance==2*M);
}
int main(void) {returns();spatial();pressure();bounds_isolation();return 0;}
'''


WRAPPER_PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include "uvm_prefetch_refault.h"
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
typedef struct {struct {NvU64 size;} mem_info;} uvm_gpu_t;
typedef struct {int list_lock;struct {uvm_prefetch_refault_t refault;} eager_eviction;} uvm_pmm_gpu_t;
static uvm_gpu_t gpu={.mem_info={.size=64ULL<<20}};
static uvm_gpu_t *uvm_pmm_to_gpu(uvm_pmm_gpu_t *p) {(void)p;return &gpu;}
static void uvm_spin_lock(int *l) {assert(!*l);*l=1;}
static void uvm_spin_unlock(int *l) {assert(*l);*l=0;}
static bool uvm_perf_prefetch_refault_enable=true;
static bool counter_admission;
static bool uvm_cpu_block_policy_counter_admission_enabled(void) { return counter_admission; }
static unsigned uvm_perf_prefetch_refault_short_percent=25;
static unsigned uvm_perf_prefetch_refault_max_neighbors=1;
'''
WRAPPER_CASES = r'''
int main(void) {
    uvm_pmm_gpu_t p={0};uvm_prefetch_refault_block_t block[32]={0};
    assert(uvm_pmm_gpu_refault_enabled());
    uvm_pmm_gpu_refault_evict(&p,&block[0]);
    uvm_pmm_gpu_refault_demand(&p,&block[0],7,0,4096,0,0,false,true);
    assert(p.eager_eviction.refault.demand_bytes==4096 && !p.eager_eviction.refault.refaults);
    uvm_pmm_gpu_refault_demand(&p,&block[0],7,0,4096,0,0,true,false);
    assert(block[0].evicted && !p.eager_eviction.refault.refaults);
    uvm_pmm_gpu_refault_demand(&p,&block[0],7,0,4096,0,1,true,true);
    assert(!block[0].evicted && p.eager_eviction.refault.last_distance==8192);
    assert(p.eager_eviction.refault.refaults==1);
    // Copying a neighbor never advances the demand-byte clock or earns a credit.
    NvU64 clock=p.eager_eviction.refault.demand_bytes;
    uvm_pmm_gpu_refault_population(&p,7,2ULL<<20);
    assert(p.eager_eviction.refault.demand_bytes==clock);
    for(unsigned i=1;i<8;i++)
        uvm_pmm_gpu_refault_demand(&p,&block[i],7,(NvU64)i<<21,0,0,i,true,false);
    assert(uvm_pmm_gpu_refault_neighbors(&p,7,0)==1);
    assert(!p.eager_eviction.refault.growths); // Default cap contains heuristic growth.
    uvm_perf_prefetch_refault_max_neighbors=4;
    for(unsigned i=8;i<16;i++)
        uvm_pmm_gpu_refault_demand(&p,&block[i],7,(NvU64)i<<21,0,0,i,true,false);
    assert(uvm_pmm_gpu_refault_neighbors(&p,7,0)==4);
    assert(p.eager_eviction.refault.growths==2);
    uvm_perf_prefetch_refault_short_percent=0;
    assert(prefetch_refault_config(&p).short_distance_percent==1);
    uvm_perf_prefetch_refault_short_percent=999;
    assert(prefetch_refault_config(&p).short_distance_percent==100);
    uvm_perf_prefetch_refault_max_neighbors=999;
    assert(prefetch_refault_config(&p).max_neighbors==31);
    uvm_perf_prefetch_refault_max_neighbors=0;
    assert(uvm_pmm_gpu_refault_neighbors(&p,7,0)==0);
    uvm_perf_prefetch_refault_enable=false;
    assert(!uvm_pmm_gpu_refault_enabled());
    uvm_pmm_gpu_refault_evict(&p,&block[0]);
    uvm_pmm_gpu_refault_demand(&p,&block[0],7,0,4096,0,99,true,true);
    uvm_pmm_gpu_refault_population(&p,7,0);
    assert(p.eager_eviction.refault.demand_bytes==clock && !block[0].evicted && !p.list_lock);
    uvm_perf_prefetch_refault_enable=true;
    counter_admission=true;
    assert(!uvm_pmm_gpu_refault_enabled());
    uvm_pmm_gpu_refault_evict(&p,&block[0]);
    uvm_pmm_gpu_refault_demand(&p,&block[0],7,0,4096,0,100,true,true);
    assert(p.eager_eviction.refault.demand_bytes==clock && !block[0].evicted);
    return 0;
}
'''


def test_refault_policy_compiles_and_runs():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'refault.c'
        exe = Path(td) / 'refault'
        source=(ROOT / 'kernel-open/nvidia-uvm/uvm_pmm_gpu.c').read_text()
        start=source.index('bool uvm_pmm_gpu_refault_enabled(')
        end=source.index('NvU16 uvm_pmm_gpu_prefetch_pattern(',start)
        wrapper=WRAPPER_PRELUDE+source[start:end]+WRAPPER_CASES
        for harness in (HARNESS,wrapper):
            src.write_text(harness)
            compile_and_run(src,exe)


def compile_and_run(src,exe):
    subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                    '-I', str(ROOT / 'kernel-open/common/inc'),
                    '-I', str(ROOT / 'kernel-open/nvidia-uvm'), str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    test_refault_policy_compiles_and_runs()
    print('Refault admission: byte distance, pressure expiry, spatial growth and identity isolation passed.')
