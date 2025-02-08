//
// Created by Mason on 2025/2/8.
//

#pragma once

#include <data/core/bi_helpers.hpp>
#include <cpu/kernels/dequantize/generic/neon/fp16.hpp>
#include <cpu/kernels/dequantize/generic/neon/fp32.hpp>

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_DEQUANTIZE_KERNEL(func_name) void func_name(const BIITensor *input, BIITensor *output, const BIWindow &window)

        DECLARE_DEQUANTIZE_KERNEL(fp32_run_dequantization_core);

        DECLARE_DEQUANTIZE_KERNEL(fp16_run_dequantization_core);

#undef DECLARE_DEQUANTIZE_KERNEL
    }
}
