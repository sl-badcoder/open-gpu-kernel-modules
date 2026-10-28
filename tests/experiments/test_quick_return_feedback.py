"""Extract and test the production quick-return protection helper."""

import pathlib
import re
import subprocess
import tempfile

source = pathlib.Path(__file__).parents[2] / "kernel-open/nvidia-uvm/uvm_pmm_gpu.c"
text = source.read_text()
assert "uvm_make_resident_cause_t cause" in text
assert "last_gpu_to_cpu_cause == UVM_MAKE_RESIDENT_CAUSE_EVICTION" in text
assert "quick_return_count >>= shifts" in text
va_block_source = (source.parent / "uvm_va_block.c").read_text()
assert va_block_source.count("transfer_mode,\n                                         contig_cause") == 1
assert va_block_source.count("transfer_mode,\n                                     contig_cause") == 1
match = re.search(r"static NvU64 migration_quick_return_protection_ns\(NvU32 count\)\n\{.*?\n\}", text, re.S)
assert match
helper = match.group(0)

program = f"""
#include <stdint.h>
#include <stdio.h>
#define NvU64 uint64_t
#define NvU32 uint32_t
#define U64_MAX UINT64_MAX
#define NSEC_PER_MSEC 1000000ULL
#define min(a,b) ((a)<(b)?(a):(b))
static unsigned uvm_perf_migration_quick_return_protection_ms;
static unsigned uvm_perf_migration_quick_return_max_protection_ms;
{helper}
int main(void) {{
  uvm_perf_migration_quick_return_protection_ms = 250;
  uvm_perf_migration_quick_return_max_protection_ms = 2000;
  if (migration_quick_return_protection_ns(0) != 0) return 1;
  if (migration_quick_return_protection_ns(1) != 250000000ULL) return 2;
  if (migration_quick_return_protection_ns(8) != 2000000000ULL) return 3;
  if (migration_quick_return_protection_ns(100) != 2000000000ULL) return 4;
  uvm_perf_migration_quick_return_protection_ms = 0;
  if (migration_quick_return_protection_ns(1) != 0) return 5;
  return 0;
}}
"""

with tempfile.TemporaryDirectory() as directory:
    c = pathlib.Path(directory) / "test.c"
    binary = pathlib.Path(directory) / "test"
    c.write_text(program)
    subprocess.run(["cc", "-std=c11", "-Wall", "-Werror", str(c), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

print("Passed: extracted production quick-return protection helper boundaries.")
