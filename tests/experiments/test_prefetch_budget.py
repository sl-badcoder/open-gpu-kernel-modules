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
static const uvm_prefetch_budget_config_t cfg = {16 * Q, 0, 20, 50, 32, true, false};
static NvU64 record_completed(uvm_prefetch_budget_t *b, const uvm_prefetch_budget_config_t *config,
                               NvU64 now, NvU64 generation)
{
    uvm_prefetch_reservation_t ticket={.generation=generation};
    NvU64 id=uvm_prefetch_budget_record(b,config,now,&ticket);
    uvm_prefetch_budget_complete(b,config,&ticket,0,true,now);
    return id;
}
static NvU64 populate_config(uvm_prefetch_budget_t *b, const uvm_prefetch_budget_config_t *config,
                             NvU64 now, unsigned n, unsigned returns)
{
    NvU64 id=0;
    for (unsigned i=0; i<n; ++i) {
        NvU64 next=record_completed(b,config,now,b->generation);
        assert(next && (!id || next==id)); id=next;
    }
    for (unsigned i=0; i<returns; ++i)
        uvm_prefetch_budget_return(b,config,id,now+1);
    return id;
}
static NvU64 populate(uvm_prefetch_budget_t *b, NvU64 now, unsigned n, unsigned returns)
{
    return populate_config(b,&cfg,now,n,returns);
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
    assert(!record_completed(&b,&cfg,100,0)); // Old in-flight reservation.
    assert(b.discarded_blocks==65);
    NvU64 fresh=populate(&b,100,32,0);
    assert(fresh!=old && fresh!=oldest);
    uvm_prefetch_budget_return(&b,&cfg,old,101);
    assert(b.dropped_returns==1 && b.quick_returns==48);
    uvm_prefetch_budget_advance(&b,&cfg,160);
    assert(b.ratio_evaluations==2 && b.last_returns==0 && b.backoffs==1);

    // Recreate a cohort in the exact same tick after a policy change.
    fresh=populate(&b,180,1,0);
    uvm_prefetch_budget_backoff(&b,&cfg,0);
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
    assert(!record_completed(&b,&cfg,0,b.generation));
    assert(b.dropped_records==1);
    uvm_prefetch_budget_advance(&b,&cfg,1000000);
    assert(b.pending_blocks==2 && b.pending_returns==1 && !b.ratio_evaluations);
}
static void reservations_pressure_and_no_growth(void)
{
    uvm_prefetch_budget_t b={0};
    uvm_prefetch_reservation_t a,c,denied;
    uvm_prefetch_budget_config_t cap={5*Q+1,Q,20,50,32,true,false};
    assert(uvm_prefetch_budget_reserve(&b,&cap,3*Q+1,64*Q,0,&a)==3*Q);
    assert(uvm_prefetch_budget_reserve(&b,&cap,3*Q,64*Q,1,&c)==2*Q);
    assert(!uvm_prefetch_budget_reserve(&b,&cap,Q,64*Q,2,&denied));
    assert(b.reserved_bytes==5*Q && b.peak_bytes==5*Q && !denied.bytes);
    uvm_prefetch_budget_backoff(&b,&cap,0); // Reserved work exceeds new cap; no underflow.
    assert(b.window_bytes==4*Q && b.generation==1);
    assert(!uvm_prefetch_budget_reserve(&b,&cap,Q,64*Q,3,&denied));
    assert(!record_completed(&b,&cap,4,a.generation));
    uvm_prefetch_budget_complete(&b,&cap,&a,3*Q,true,0);
    uvm_prefetch_budget_complete(&b,&cap,&c,0,true,0);
    assert(!b.reserved_bytes && b.window_bytes==4*Q);
    uvm_prefetch_budget_advance(&b,&cap,1000000);
    assert(b.window_bytes==4*Q); // Completion and quiet time never grow the window.
    assert(!uvm_prefetch_budget_reserve(&b,&cap,Q,Q,1000001,&denied));
    assert(b.pressure_pauses==1 && b.capacity_stops==1 && b.backoffs==1);
    assert(uvm_prefetch_budget_reserve(&b,&cap,2*Q,2*Q,1000002,&a)==Q);
    uvm_prefetch_budget_complete(&b,&cap,&a,Q,true,0);
    cap.limit_bytes=Q-1;
    assert(!uvm_prefetch_budget_reserve(&b,&cap,Q,64*Q,1000003,&denied));
}
static void permanent_off_and_gentle_steps(void)
{
    uvm_prefetch_budget_t b={0};
    const unsigned expected[]={12,9,7,6,5,4,3,2,0};
    for (unsigned i=0;i<sizeof(expected)/sizeof(expected[0]);++i) {
        uvm_prefetch_budget_backoff(&b,&cfg,0);
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
        id=record_completed(&b,&disabled,0,ticket.generation);
    for(unsigned i=0;i<32;++i)
        uvm_prefetch_budget_return(&b,&disabled,id,1);
    uvm_prefetch_budget_advance(&b,&disabled,60);
    assert(b.ratio_evaluations==1 && b.last_blocks==32 && b.last_returns==32);
    assert(!b.backoffs && !b.generation && b.window_bytes==16*Q);
    assert(b.reserved_bytes==4*Q); // Existing work remains accounted for.
    uvm_prefetch_budget_complete(&b,&disabled,&ticket,4*Q,true,1);
    assert(uvm_prefetch_budget_reserve(&b,&disabled,16*Q,64*Q,80,&ticket)==16*Q);
    uvm_prefetch_budget_complete(&b,&disabled,&ticket,16*Q,true,81);
    assert(!uvm_prefetch_budget_reserve(&b,&disabled,Q,0,81,&ticket));
    assert(b.pressure_pauses==1 && !b.backoffs); // Headroom safety still applies.
    assert(b.window_bytes==16*Q && b.quick_returns==32);
}
static void active_recovery_needs_mature_fresh_ratio(void)
{
    uvm_prefetch_budget_config_t recovery=cfg;
    recovery.recovery_enabled=true;
    uvm_prefetch_budget_t b={.initialized=true,.window_bytes=4*Q};
    uvm_prefetch_reservation_t old;
    assert(uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,0,&old)==2*Q);
    populate_config(&b,&recovery,19,32,15);
    NvU64 immature=populate_config(&b,&recovery,40,10,0);
    uvm_prefetch_budget_advance(&b,&recovery,59);
    assert(b.window_bytes==4*Q && !b.recoveries);
    uvm_prefetch_budget_advance(&b,&recovery,60);
    assert(b.window_bytes==5*Q && b.recoveries==1 && b.generation==1);
    assert(b.last_blocks==32 && b.last_returns==15 && b.discarded_blocks==10);
    assert(!record_completed(&b,&recovery,61,old.generation));
    uvm_prefetch_budget_return(&b,&recovery,immature,61);
    assert(b.dropped_returns==1 && b.discarded_blocks==11);
    uvm_prefetch_budget_complete(&b,&recovery,&old,2*Q,true,62);
    uvm_prefetch_budget_advance(&b,&recovery,1000000);
    assert(b.recoveries==1 && b.window_bytes==5*Q); // Time/completion cannot grow it.
    populate_config(&b,&recovery,1000020,31,0);
    uvm_prefetch_budget_advance(&b,&recovery,1000080);
    assert(b.recoveries==1 && b.pending_blocks==31);
    populate_config(&b,&recovery,1000080,1,0);
    uvm_prefetch_budget_advance(&b,&recovery,1000140);
    assert(b.recoveries==2 && b.window_bytes==6*Q);
    populate_config(&b,&recovery,1000140,32,16);
    uvm_prefetch_budget_advance(&b,&recovery,1000200);
    assert(b.backoffs==1 && b.recoveries==2 && b.window_bytes==5*Q); // Equality backs off.
}
static void recovery_respects_cap_and_switches(void)
{
    uvm_prefetch_budget_config_t recovery=cfg;
    recovery.recovery_enabled=true;
    uvm_prefetch_budget_t b={.initialized=true,.window_bytes=15*Q};
    populate_config(&b,&recovery,0,32,0);
    uvm_prefetch_budget_advance(&b,&recovery,60);
    assert(b.window_bytes==16*Q && b.recoveries==1);
    populate_config(&b,&recovery,60,32,0);
    uvm_prefetch_budget_advance(&b,&recovery,120);
    assert(b.window_bytes==16*Q && b.recoveries==1 && b.generation==1);
    b.window_bytes=4*Q;
    recovery.backoff_enabled=false;
    populate_config(&b,&recovery,120,32,0);
    uvm_prefetch_budget_advance(&b,&recovery,180);
    assert(b.window_bytes==4*Q && b.recoveries==1);
    recovery.backoff_enabled=true;
    recovery.recovery_enabled=false;
    populate_config(&b,&recovery,180,32,0);
    uvm_prefetch_budget_advance(&b,&recovery,240);
    assert(b.window_bytes==4*Q && b.recoveries==1);
}
static void off_probes_require_real_ratio_evidence(void)
{
    for(unsigned returns=15;returns<=16;++returns) {
        uvm_prefetch_budget_config_t recovery=cfg;
        recovery.recovery_enabled=true;
        uvm_prefetch_budget_t b={.initialized=true,.window_bytes=2*Q};
        uvm_prefetch_reservation_t probe,denied;
        uvm_prefetch_budget_backoff(&b,&recovery,0);
        assert(!b.window_bytes && b.next_probe_ns==60);
        assert(!uvm_prefetch_budget_reserve(&b,&recovery,4*Q,64*Q,59,&denied));
        assert(!uvm_prefetch_budget_reserve(&b,&recovery,Q,64*Q,60,&denied));
        assert(!uvm_prefetch_budget_reserve(&b,&recovery,4*Q,Q,60,&denied));
        NvU64 now=60;
        // Each probe copies one neighbor. Do not reopen after one clean probe:
        // all 32 neighbors must mature, exactly as for ordinary decisions.
        for(unsigned i=0;i<32;++i) {
            assert(uvm_prefetch_budget_reserve(&b,&recovery,8*Q,64*Q,now,&probe)==2*Q);
            assert(probe.probe && b.probe_inflight && !b.window_bytes);
            assert(!uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,now+1,&denied));
            NvU64 id=uvm_prefetch_budget_record(&b,&recovery,now,&probe);
            assert(id);
            if(i<returns) uvm_prefetch_budget_return(&b,&recovery,id,now+1);
            uvm_prefetch_budget_complete(&b,&recovery,&probe,2*Q,true,now+2);
            assert(!b.probe_inflight && !b.window_bytes && !b.recoveries);
            if(i<31)
                assert(!uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,now+61,&denied));
            now+=62;
        }
        assert(uvm_prefetch_budget_reserve(&b,&recovery,8*Q,64*Q,now,&probe)==2*Q);
        assert(b.last_blocks==32 && b.last_returns==returns && b.probes==(returns==15?32:33));
        assert(b.window_bytes==(returns==15?2*Q:0) && b.recoveries==(returns==15));
        assert(probe.probe==(returns==16));
        uvm_prefetch_budget_complete(&b,&recovery,&probe,0,true,now+1);
    }
}
static void probe_empty_drain_and_small_cap(void)
{
    uvm_prefetch_budget_config_t recovery=cfg;
    recovery.recovery_enabled=true;
    uvm_prefetch_budget_t b={.initialized=true,.window_bytes=2*Q};
    uvm_prefetch_reservation_t old,probe,denied;
    assert(uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,0,&old)==2*Q);
    uvm_prefetch_budget_backoff(&b,&recovery,1);
    assert(!uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,100,&denied));
    uvm_prefetch_budget_complete(&b,&recovery,&old,2*Q,true,101);
    for(unsigned i=0;i<4;++i) {
        NvU64 now=102+i*61;
        assert(uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,now,&probe)==2*Q);
        uvm_prefetch_budget_complete(&b,&recovery,&probe,0,true,now+1);
        assert(!b.window_bytes && !b.recoveries && !b.ratio_evaluations);
    }
    recovery.limit_bytes=Q;
    assert(!uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,1000000,&denied));
    assert(!b.probe_inflight && b.probes==4);
}
static void incomplete_and_failed_attempts_cannot_recover(void)
{
    uvm_prefetch_budget_config_t recovery=cfg;
    recovery.recovery_enabled=true; recovery.min_blocks=2;
    uvm_prefetch_budget_t b={.initialized=true,.window_bytes=4*Q};
    uvm_prefetch_reservation_t ticket;
    assert(uvm_prefetch_budget_reserve(&b,&recovery,3*Q,64*Q,0,&ticket)==3*Q);
    assert(uvm_prefetch_budget_record(&b,&recovery,1,&ticket));
    assert(uvm_prefetch_budget_record(&b,&recovery,1,&ticket));
    uvm_prefetch_budget_advance(&b,&recovery,100);
    assert(!b.mature_blocks && !b.ratio_evaluations && !b.recoveries);
    // Mapping, dependency, or final GPU wait failure invalidates its entire cohort.
    uvm_prefetch_budget_complete(&b,&recovery,&ticket,2*Q,false,101);
    assert(b.discarded_blocks==2 && !b.mature_blocks && !b.recoveries && !b.reserved_bytes);
    assert(b.window_bytes==4*Q && !b.generation);
    assert(uvm_prefetch_budget_reserve(&b,&recovery,3*Q,64*Q,120,&ticket)==3*Q);
    assert(uvm_prefetch_budget_record(&b,&recovery,121,&ticket));
    assert(uvm_prefetch_budget_record(&b,&recovery,121,&ticket));
    // If work completes late but successfully, completed evidence can mature now.
    uvm_prefetch_budget_complete(&b,&recovery,&ticket,2*Q,true,200);
    assert(b.recoveries==1 && b.window_bytes==5*Q && b.last_blocks==2);
}
static void concurrent_failure_and_same_tick_replacement(void)
{
    uvm_prefetch_budget_config_t recovery=cfg;
    recovery.recovery_enabled=true; recovery.min_blocks=2;
    uvm_prefetch_budget_t b={.initialized=true,.window_bytes=8*Q};
    uvm_prefetch_reservation_t a,failed,fresh;
    assert(uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,0,&a)==2*Q);
    assert(uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,0,&failed)==2*Q);
    NvU64 old=uvm_prefetch_budget_record(&b,&recovery,1,&a);
    assert(uvm_prefetch_budget_record(&b,&recovery,1,&failed)==old);
    uvm_prefetch_budget_complete(&b,&recovery,&failed,Q,false,2);
    assert(b.discarded_blocks==2 && !b.mature_blocks);
    assert(uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,3,&fresh)==2*Q);
    NvU64 replacement=uvm_prefetch_budget_record(&b,&recovery,4,&fresh);
    assert(replacement && replacement!=old);
    // The still-live ticket cannot overwrite its previous reference after failure.
    assert(!uvm_prefetch_budget_record(&b,&recovery,5,&a));
    uvm_prefetch_budget_return(&b,&recovery,old,6);
    assert(b.dropped_records==1 && b.dropped_returns==1);
    uvm_prefetch_budget_complete(&b,&recovery,&a,Q,true,70);
    assert(!b.mature_blocks); // Fresh contribution still awaiting completion.
    uvm_prefetch_budget_complete(&b,&recovery,&fresh,Q,true,71);
    assert(b.mature_blocks==1 && b.pending_blocks==1 && !b.recoveries && !b.reserved_bytes);
}
static void failed_probe_does_not_help_reopen(void)
{
    uvm_prefetch_budget_config_t recovery=cfg;
    recovery.recovery_enabled=true; recovery.min_blocks=2;
    uvm_prefetch_budget_t b={.initialized=true};
    uvm_prefetch_reservation_t probe;
    for(unsigned i=0;i<3;++i) {
        NvU64 now=80*i;
        assert(uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,now,&probe)==2*Q);
        assert(probe.probe);
        assert(uvm_prefetch_budget_record(&b,&recovery,now+1,&probe));
        uvm_prefetch_budget_complete(&b,&recovery,&probe,Q,i!=0,now+2);
        uvm_prefetch_budget_advance(&b,&recovery,now+60);
        assert(b.recoveries==(i==2));
    }
    assert(b.discarded_blocks==1 && b.last_blocks==2 && b.last_returns==0);
    assert(b.window_bytes==2*Q && !b.probe_inflight && !b.reserved_bytes);
}
static void configured_percentage_controls_both_directions(void)
{
    const unsigned percentages[]={50,75,100};
    for(unsigned p=0;p<3;++p) {
        for(unsigned equality=0;equality<2;++equality) {
            uvm_prefetch_budget_config_t recovery=cfg;
            recovery.recovery_enabled=true; recovery.return_percent=percentages[p];
            uvm_prefetch_budget_t b={.initialized=true,.window_bytes=4*Q};
            unsigned threshold=32*percentages[p]/100;
            populate_config(&b,&recovery,0,32,threshold-1+equality);
            uvm_prefetch_budget_advance(&b,&recovery,60);
            assert(b.window_bytes==(equality?3:5)*Q);
            assert(b.backoffs==equality && b.recoveries==!equality);
        }
    }
}
static void shared_cohort_failure_in_both_completion_orders(void)
{
    for(unsigned fail_first=0;fail_first<2;++fail_first) {
        uvm_prefetch_budget_config_t recovery=cfg;
        recovery.recovery_enabled=true; recovery.min_blocks=2;
        uvm_prefetch_budget_t b={.initialized=true,.window_bytes=8*Q};
        uvm_prefetch_reservation_t good,bad;
        assert(uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,0,&good)==2*Q);
        assert(uvm_prefetch_budget_reserve(&b,&recovery,2*Q,64*Q,0,&bad)==2*Q);
        assert(uvm_prefetch_budget_record(&b,&recovery,1,&good));
        assert(uvm_prefetch_budget_record(&b,&recovery,1,&bad));
        if(fail_first) uvm_prefetch_budget_complete(&b,&recovery,&bad,Q,false,2);
        else uvm_prefetch_budget_complete(&b,&recovery,&good,Q,true,2);
        uvm_prefetch_budget_advance(&b,&recovery,60);
        assert(!b.ratio_evaluations && !b.recoveries);
        if(fail_first) uvm_prefetch_budget_complete(&b,&recovery,&good,Q,true,61);
        else uvm_prefetch_budget_complete(&b,&recovery,&bad,Q,false,61);
        assert(b.discarded_blocks==2 && !b.mature_blocks && !b.reserved_bytes);
        assert(b.window_bytes==8*Q && !b.generation && !b.recoveries);
    }
}
int main(void)
{
    configured_percentage_controls_both_directions(); shared_cohort_failure_in_both_completion_orders();
    incomplete_and_failed_attempts_cannot_recover(); concurrent_failure_and_same_tick_replacement();
    failed_probe_does_not_help_reopen();
    active_recovery_needs_mature_fresh_ratio(); recovery_respects_cap_and_switches();
    off_probes_require_real_ratio_evidence(); probe_empty_drain_and_small_cap();
    disabled_backoff_preserves_prefetch_and_measurement();
    threshold_and_maturity(); minimum_and_fresh_evidence(); new_policy_requires_new_evidence();
    expired_tokens_and_delayed_records(); reservations_pressure_and_no_growth(); permanent_off_and_gentle_steps();
    puts("Rapid-return ratio: thresholds, maturity, epochs, caps, pressure, ratio recovery, bounded OFF probes and recovery disable passed.");
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
