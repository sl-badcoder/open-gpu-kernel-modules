/* SPDX-License-Identifier: MIT */
#ifndef __UVM_PERF_DIAGNOSTICS_H__
#define __UVM_PERF_DIAGNOSTICS_H__

#include "uvm_common.h"

typedef enum
{
    UVM_PERF_DIAG_COUNTER_SERVICE,
    UVM_PERF_DIAG_POPULATION_SUBMIT,
    UVM_PERF_DIAG_POPULATION_WAIT,
    UVM_PERF_DIAG_EVICTION_SELECT,
    UVM_PERF_DIAG_EVICTION_SUBMIT,
    UVM_PERF_DIAG_EVICTION_DRAIN,
    UVM_PERF_DIAG_COUNTER_BLOCK,
    UVM_PERF_DIAG_STAGE_COUNT
} uvm_perf_diag_stage_t;

extern bool uvm_perf_placement_diagnostics_enable;
void uvm_perf_diag_record(uvm_perf_diag_stage_t stage, NvU64 elapsed_ns);
void uvm_perf_diag_population(NvU64 bytes, bool success);

// Read-only load-time switch. Disabled hooks perform no clock reads or updates.
static inline NvU64 uvm_perf_diag_begin(void)
{
    return uvm_perf_placement_diagnostics_enable ? NV_GETTIME() : 0;
}

static inline void uvm_perf_diag_end(uvm_perf_diag_stage_t stage, NvU64 start)
{
    if (start)
        uvm_perf_diag_record(stage, NV_GETTIME() - start);
}

#endif
