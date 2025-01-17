//
// Created by Mason on 2025/1/17.
//

#pragma once

#include <cpu/kernels/mul/generic/neon/fp16.hpp>
#include <cpu/kernels/mul/generic/neon/fp32.hpp>

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_MUL_KERNEL(func_name) \
    void func_name(const BIITensor *src1, const BIITensor *src2, BIITensor *out, const BIWindow &window, float scale)

        DECLARE_MUL_KERNEL(mul_F32_F32_F32);

        DECLARE_MUL_KERNEL(mul_F16_F16_F16);

#undef DECLARE_MUL_KERNEL
    }
} // namespace BatmanInfer
