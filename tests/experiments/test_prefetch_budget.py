#!/usr/bin/env python3
"""Deterministic tests for the rapid-return ratio controller."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).parents[2]
PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef uint64_t NvU64;
#define UVM_ASSERT assert
'''
CASES = r'''
#define Q UVM_PREFETCH_BUDGET_QUANTUM
static const uvm_prefetch_budget_config_t cfg = {16 * Q, 0, 20, 50, 32, true};
static NvU64 populate(uvm_prefetch_budget_t *b, NvU64 now, unsigned n, unsigned returns)
{
    NvU64 id=0;
    for (unsigned i=0; i<n; ++i) {
        NvU64 next=uvm_prefetch_budget_record(b,&cfg,now,b->generation);
        assert(next && (!id || next==id)); id=next;
    }
    for (unsigned i=0; i<returns; ++i)
        uvm_prefetch_budget_return(b,&cfg,id,now+1);
    return id;
}
static void threshold_and_maturity(void)
{
    for (unsigned returns=15; returns<=16; ++returns) {
        uvm_prefetch_budget_t b={.initialized=true,.window_bytes=16*Q};
        populate(&b,19,32,returns);
        uvm_prefetch_budget_advance(&b,&cfg,59);
        assert(!b.ratio_evaluations && !b.mature_blocks); // Last member gets full 40ms.
        uvm_prefetch_budget_advance(&b,&cfg,60);
        assert(b.ratio_evaluations==1 && b.last_blocks==32 && b.last_returns==returns);
        assert(b.backoffs==(returns==16) && b.window_bytes==(returns==16?12:16)*Q);
        uvm_prefetch_budget_advance(&b,&cfg,1000000);
        assert(b.ratio_evaluations==1); // Consumed evidence cannot repeat.
    }
    uvm_prefetch_budget_t b={0};
    populate(&b,0,33,16);
    uvm_prefetch_budget_advance(&b,&cfg,60);
    assert(!b.backoffs); // 16/33 < 50%, do not truncate the percentage.
    populate(&b,80,33,17);
    uvm_prefetch_budget_advance(&b,&cfg,140);
    assert(b.backoffs==1 && b.last_returns==17 && b.last_blocks==33);
}
static void minimum_and_fresh_evidence(void)
{
    uvm_prefetch_budget_t b={0};
    populate(&b,0,16,8);
    uvm_prefetch_budget_advance(&b,&cfg,60);
    assert(!b.ratio_evaluations && b.pending_blocks==16 && b.pending_returns==8);
    populate(&b,60,16,8);
    uvm_prefetch_budget_advance(&b,&cfg,120);
    assert(b.ratio_evaluations==1 && b.backoffs==1 && b.last_blocks==32);
    assert(!b.pending_blocks && !b.pending_returns);

    memset(&b,0,sizeof(b));
    populate(&b,0,32,16);
    populate(&b,40,200,0);
    uvm_prefetch_budget_advance(&b,&cfg,60);
    assert(b.backoffs==1 && b.last_blocks==32 && b.last_returns==16);
    assert(b.discarded_blocks==200); // Fresh populations did not dilute the ratio.
}
static void new_policy_requires_new_evidence(void)
{
    uvm_prefetch_budget_t b={0};
    NvU64 oldest=populate(&b,0,32,16);
    NvU64 old=populate(&b,20,32,16);
    populate(&b,40,32,16);
    uvm_prefetch_budget_advance(&b,&cfg,60);
    assert(b.backoffs==1 && b.generation==1 && b.discarded_blocks==64);
    assert(b.discarded_returns==32 && b.window_bytes==12*Q);
    uvm_prefetch_budget_advance(&b,&cfg,100);
    assert(b.backoffs==1 && b.ratio_evaluations==1);
    assert(!uvm_prefetch_budget_record(&b,&cfg,100,0)); // Old in-flight reservation.
    assert(b.discarded_blocks==65);
    NvU64 fresh=populate(&b,100,32,0);
    assert(fresh!=old && fresh!=oldest);
    uvm_prefetch_budget_return(&b,&cfg,old,101);
    assert(b.dropped_returns==1 && b.quick_returns==48);
    uvm_prefetch_budget_advance(&b,&cfg,160);
    assert(b.ratio_evaluations==2 && b.last_returns==0 && b.backoffs==1);

    // Recreate a cohort in the exact same tick after a policy change.
    fresh=populate(&b,180,1,0);
    uvm_prefetch_budget_backoff(&b,&cfg);
    NvU64 replacement=populate(&b,180,1,0);
    assert(fresh!=replacement);
    uvm_prefetch_budget_return(&b,&cfg,fresh,181);
    assert(b.dropped_returns==2);
}
static void expired_tokens_and_delayed_records(void)
{
    uvm_prefetch_budget_t b={0};
    NvU64 old=populate(&b,0,1,1);
    uvm_prefetch_budget_return(&b,&cfg,old,2); // Cannot exceed recorded populations.
    assert(b.quick_returns==1);
    populate(&b,80,1,0); // Same modulo slot, unique ID.
    uvm_prefetch_budget_return(&b,&cfg,old,81);
    assert(b.quick_returns==1 && b.dropped_returns==2);
    assert(!uvm_prefetch_budget_record(&b,&cfg,0,b.generation));
    assert(b.dropped_records==1);
    uvm_prefetch_budget_advance(&b,&cfg,1000000);
    assert(b.pending_blocks==2 && b.pending_returns==1 && !b.ratio_evaluations);
}
static void reservations_pressure_and_no_growth(void)
{
    uvm_prefetch_budget_t b={0};
    uvm_prefetch_reservation_t a,c,denied;
    uvm_prefetch_budget_config_t cap={5*Q+1,Q,20,50,32,true};
    assert(uvm_prefetch_budget_reserve(&b,&cap,3*Q+1,64*Q,0,&a)==3*Q);
    assert(uvm_prefetch_budget_reserve(&b,&cap,3*Q,64*Q,1,&c)==2*Q);
    assert(!uvm_prefetch_budget_reserve(&b,&cap,Q,64*Q,2,&denied));
    assert(b.reserved_bytes==5*Q && b.peak_bytes==5*Q && !denied.bytes);
    uvm_prefetch_budget_backoff(&b,&cap); // Reserved work exceeds new cap; no underflow.
    assert(b.window_bytes==4*Q && b.generation==1);
    assert(!uvm_prefetch_budget_reserve(&b,&cap,Q,64*Q,3,&denied));
    assert(!uvm_prefetch_budget_record(&b,&cap,4,a.generation));
    uvm_prefetch_budget_complete(&b,&a,3*Q);
    uvm_prefetch_budget_complete(&b,&c,0);
    assert(!b.reserved_bytes && b.window_bytes==4*Q);
    uvm_prefetch_budget_advance(&b,&cap,1000000);
    assert(b.window_bytes==4*Q); // Completion and quiet time never grow the window.
    assert(!uvm_prefetch_budget_reserve(&b,&cap,Q,Q,1000001,&denied));
    assert(b.pressure_pauses==1 && b.capacity_stops==1 && b.backoffs==1);
    assert(uvm_prefetch_budget_reserve(&b,&cap,2*Q,2*Q,1000002,&a)==Q);
    uvm_prefetch_budget_complete(&b,&a,Q);
    cap.limit_bytes=Q-1;
    assert(!uvm_prefetch_budget_reserve(&b,&cap,Q,64*Q,1000003,&denied));
}
static void permanent_off_and_gentle_steps(void)
{
    uvm_prefetch_budget_t b={0};
    const unsigned expected[]={12,9,7,6,5,4,3,2,0};
    for (unsigned i=0;i<sizeof(expected)/sizeof(expected[0]);++i) {
        uvm_prefetch_budget_backoff(&b,&cfg);
        assert(b.window_bytes==expected[i]*Q && b.backoffs==i+1);
    }
    uvm_prefetch_reservation_t denied;
    for (unsigned i=0;i<4;++i) {
        assert(!uvm_prefetch_budget_reserve(&b,&cfg,16*Q,64*Q,1000000ULL<<i,&denied));
        assert(!b.window_bytes && !denied.bytes && b.backoffs==9);
    }
}
static void disabled_backoff_preserves_prefetch_and_measurement(void)
{
    uvm_prefetch_budget_t b={0};
    uvm_prefetch_budget_config_t disabled=cfg;
    uvm_prefetch_reservation_t ticket;
    disabled.backoff_enabled=false;
    assert(uvm_prefetch_budget_reserve(&b,&disabled,4*Q,64*Q,0,&ticket)==4*Q);
    NvU64 id=0;
    for(unsigned i=0;i<32;++i)
        id=uvm_prefetch_budget_record(&b,&disabled,0,ticket.generation);
    for(unsigned i=0;i<32;++i)
        uvm_prefetch_budget_return(&b,&disabled,id,1);
    uvm_prefetch_budget_advance(&b,&disabled,60);
    assert(b.ratio_evaluations==1 && b.last_blocks==32 && b.last_returns==32);
    assert(!b.backoffs && !b.generation && b.window_bytes==16*Q);
    assert(b.reserved_bytes==4*Q); // Existing work remains accounted for.
    uvm_prefetch_budget_complete(&b,&ticket,4*Q);
    assert(uvm_prefetch_budget_reserve(&b,&disabled,16*Q,64*Q,80,&ticket)==16*Q);
    uvm_prefetch_budget_complete(&b,&ticket,16*Q);
    assert(!uvm_prefetch_budget_reserve(&b,&disabled,Q,0,81,&ticket));
    assert(b.pressure_pauses==1 && !b.backoffs); // Headroom safety still applies.
    assert(b.window_bytes==16*Q && b.quick_returns==32);
}
int main(void)
{
    disabled_backoff_preserves_prefetch_and_measurement();
    threshold_and_maturity(); minimum_and_fresh_evidence(); new_policy_requires_new_evidence();
    expired_tokens_and_delayed_records(); reservations_pressure_and_no_growth(); permanent_off_and_gentle_steps();
    puts("Rapid-return ratio: thresholds, maturity, epochs, caps, pressure, no recovery and permanent OFF passed.");
    return 0;
}
'''

def main():
    header = (ROOT / 'kernel-open/nvidia-uvm/uvm_prefetch_budget.h').read_text()
    code = PRELUDE + header.replace('#include "nvtypes.h"', '') + CASES
    with tempfile.TemporaryDirectory(prefix='prefetch-budget-') as directory:
        c = Path(directory) / 'test.c'; binary = Path(directory) / 'test'
        c.write_text(code)
        subprocess.run(['cc', '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', '-g', str(c), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
if __name__ == '__main__': main()
