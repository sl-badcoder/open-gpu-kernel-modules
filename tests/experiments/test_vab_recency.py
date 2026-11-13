#!/usr/bin/env python3
"""Run the production sampler and recency query, including lifecycle gaps."""
from pathlib import Path
import subprocess
import tempfile
import test_vab_observation as observation

CASES = r'''
 // Use the production sampler and production query; only RM/Linux are mocked.
 static uvm_parent_gpu_t q;
 q.rm_device=&q;active_parent=&q;q.access_bits_supported=true;
 q.vab_info.accessBitsBufferHandle=1;q.vab_info.granularity=6;q.vab_info.enabledMask[0]=3;
 uvm_perf_vab_interval_ms=100;uvm_perf_vab_eviction_linear_base=0;uvm_perf_vab_mode=3;
 uvm_gpu_vab_init(&q);uvm_gpu_vab_start(&q);
 dump_bits=1;NvU64 first=now;vab_sample(&q);NvU64 recency;
 uvm_gpu_vab_get_snapshot(&q,&s);assert(s.eviction_enabled&&s.lru_enabled);
 assert(uvm_gpu_vab_get_recency(&q,&s,0,2ULL<<20,first-1,&recency)&&recency==first);
 assert(uvm_gpu_vab_get_recency(&q,&s,2ULL<<20,2ULL<<20,first-1,&recency)&&recency==first);
 // A steady one must not move to MRU on every poll.
 now+=NSEC_PER_MSEC;vab_sample(&q);uvm_gpu_vab_get_snapshot(&q,&s);
 assert(uvm_gpu_vab_get_recency(&q,&s,0,2ULL<<20,first-1,&recency)&&recency==first);
 dump_bits=0;vab_sample(&q);uvm_gpu_vab_get_snapshot(&q,&s);
 assert(uvm_gpu_vab_get_recency(&q,&s,0,2ULL<<20,first-1,&recency)&&recency==first);
 // Transitions observed in mode 1 count when mode 3 is enabled later.
 uvm_perf_vab_mode=1;dump_bits=1;NvU64 second=now;vab_sample(&q);
 uvm_gpu_vab_get_snapshot(&q,&s);assert(!uvm_gpu_vab_get_recency(&q,&s,0,1,0,&recency));
 uvm_perf_vab_mode=3;uvm_gpu_vab_get_snapshot(&q,&s);
 assert(uvm_gpu_vab_get_recency(&q,&s,0,1,0,&recency)&&recency==second);
 // Readiness prevents using samples from before destination completion.
 assert(!uvm_gpu_vab_get_recency(&q,&s,0,1,s.sampled_ns+1,&recency));
 // A racing publication rejects the older snapshot.
 uvm_vab_snapshot_t old=s;vab_sample(&q);
 assert(!uvm_gpu_vab_get_recency(&q,&old,0,1,0,&recency));
 // Disabled coverage is unknown and loses history; re-enabled set bits seed.
 q.vab_info.enabledMask[0]=2;vab_sample(&q);uvm_gpu_vab_get_snapshot(&q,&s);
 assert(q.vab.last_access_ns[0]==0&&!uvm_gpu_vab_get_recency(&q,&s,0,1,0,&recency));
 q.vab_info.enabledMask[0]=3;NvU64 reseed=now;vab_sample(&q);uvm_gpu_vab_get_snapshot(&q,&s);
 assert(uvm_gpu_vab_get_recency(&q,&s,0,1,0,&recency)&&recency==reseed);
 // A layout change clears history for old bit meanings, and seeds current bits.
 q.vab_info.granularity=0;dump_bits=2;reseed=now;vab_sample(&q);uvm_gpu_vab_get_snapshot(&q,&s);
 assert(q.vab.last_access_ns[0]==0&&q.vab.last_access_ns[1]==reseed);
 assert(uvm_gpu_vab_get_recency(&q,&s,0,128ULL<<10,0,&recency)&&recency==reseed);
 assert(!uvm_gpu_vab_get_recency(&q,&s,0,192ULL<<10,0,&recency));
 // Error, dirty dump, suspend and long sampling gaps each reseed next valid data.
 dump_status=NV_ERR_INVALID_STATE;vab_sample(&q);dump_status=NV_OK;reseed=now;vab_sample(&q);
 assert(q.vab.last_access_ns[1]==reseed);
 q.vab_info.bDirtyBits=true;vab_sample(&q);assert(!q.vab.sample.valid);
 q.vab_info.bDirtyBits=false;reseed=now;vab_sample(&q);assert(q.vab.last_access_ns[1]==reseed);
 uvm_gpu_vab_invalidate(&q);reseed=now;vab_sample(&q);assert(q.vab.last_access_ns[1]==reseed);
 now+=301*NSEC_PER_MSEC;reseed=now;vab_sample(&q);assert(q.vab.last_access_ns[1]==reseed);
 // Mode zero/PM skip invalidate the window too.
 uvm_perf_vab_mode=0;vab_sample(&q);uvm_perf_vab_mode=3;reseed=now;vab_sample(&q);
 assert(q.vab.last_access_ns[1]==reseed);
 pm_allowed=false;vab_sample(&q);pm_allowed=true;reseed=now;vab_sample(&q);
 assert(q.vab.last_access_ns[1]==reseed);
 uvm_gpu_vab_get_snapshot(&q,&s);s.lru_enabled=false;
 assert(!uvm_gpu_vab_get_recency(&q,&s,0,1,0,&recency));
 // Region mode treats every set bit as recently observed, never first-set age.
 region_mode=true;dump_bits=2;NvU64 observed_now=now;vab_sample(&q);
 assert(q.vab.last_access_ns[1]==observed_now);
 observed_now=now;vab_sample(&q);assert(q.vab.last_access_ns[1]==observed_now);
 assert(observed_now>reseed);
 dump_bits=0;observed_now=q.vab.last_access_ns[1];vab_sample(&q);
 assert(q.vab.last_access_ns[1]==observed_now); // Clear bits age.
 region_mode=false;uvm_gpu_vab_stop(&q);
'''

def main():
    root=Path(__file__).parents[2]
    header=(root/'kernel-open/nvidia-uvm/uvm_vab.h').read_text().replace('#include "nvtypes.h"','')
    pmm=(root/'kernel-open/nvidia-uvm/uvm_pmm_gpu.h').read_text()
    end=pmm.index('} uvm_pmm_gpu_vab_stats_t;')+len('} uvm_pmm_gpu_vab_stats_t;')
    stats=pmm[pmm.rfind('typedef struct',0,end):end]
    source=(root/'kernel-open/nvidia-uvm/uvm_gpu_vab.c').read_text()
    source='\n'.join(line for line in source.splitlines() if not line.startswith('#include'))
    cases=observation.CASES.replace(' puts("VAB production',CASES+'\n puts("VAB production')
    with tempfile.TemporaryDirectory(prefix='vab-recency-') as directory:
        c=Path(directory)/'test.c';exe=Path(directory)/'test'
        c.write_text(observation.PRELUDE+header+stats+observation.MOCKS+source+cases)
        subprocess.run(['cc','-std=c11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(c),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)
    print('Production VAB recency transitions, mode changes, layout and lifecycle tests passed.')

if __name__=='__main__': main()
