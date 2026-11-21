"""Diagnostic hook doubles for extracted production placement tests."""

DIAGNOSTICS_STUBS = r'''
typedef enum {
 UVM_PERF_DIAG_COUNTER_SERVICE, UVM_PERF_DIAG_POPULATION_SUBMIT,
 UVM_PERF_DIAG_POPULATION_WAIT, UVM_PERF_DIAG_EVICTION_SELECT,
 UVM_PERF_DIAG_EVICTION_SUBMIT, UVM_PERF_DIAG_EVICTION_DRAIN,
 UVM_PERF_DIAG_COUNTER_BLOCK,
 UVM_PERF_DIAG_STAGE_COUNT
} uvm_perf_diag_stage_t;
static unsigned diag_calls[UVM_PERF_DIAG_STAGE_COUNT];
static unsigned diag_empty, diag_nonempty, diag_failed;
static inline __attribute__((unused)) NvU64 uvm_perf_diag_begin(void) { return 1; }
static inline __attribute__((unused)) void uvm_perf_diag_end(uvm_perf_diag_stage_t stage, NvU64 start) {
 if (start) ++diag_calls[stage];
}
static inline __attribute__((unused)) void uvm_perf_diag_population(NvU64 bytes, bool success) {
 if (bytes) ++diag_nonempty; else ++diag_empty;
 if (!success) ++diag_failed;
}
'''
