/*******************************************************************************
    Copyright (c) 2026 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#ifndef __UVM_CPU_BLOCK_POLICY_H__
#define __UVM_CPU_BLOCK_POLICY_H__

#include "uvm_common.h"
#include "uvm_forward_decl.h"

bool uvm_cpu_block_policy_enabled(void);
void uvm_cpu_block_policy_init_range(uvm_va_range_managed_t *managed_range);
bool uvm_cpu_block_policy_should_add_accessed_by(uvm_va_range_managed_t *managed_range,
                                                uvm_gpu_t *gpu);
bool uvm_cpu_block_policy_should_promote(uvm_va_block_t *va_block, uvm_gpu_t *gpu);
void uvm_cpu_block_policy_record_promotion(void);

#endif
