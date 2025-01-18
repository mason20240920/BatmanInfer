//
// Created by Mason on 2025/1/18.
//

#pragma once

#include <cpu/kernels/softmax/generic/neon/fp32.hpp>
#include <cpu/kernels/softmax/generic/neon/fp16.hpp>
#include <cpu/kernels/softmax/generic/neon/qasymm8_signed.hpp>
#include <cpu/kernels/softmax/generic/neon/qasymm8.hpp>

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_SOFTMAX_KERNEL(func_name)                                                                              \
    template <bool IS_LOG>                                                                                             \
    void func_name(const BIITensor *in, void *const tmp, BIITensor *out, const float beta, int axis, const BIWindow &window, \
                   const void *lut_ptr)

        DECLARE_SOFTMAX_KERNEL(neon_fp32_softmax);

        DECLARE_SOFTMAX_KERNEL(neon_fp16_softmax);

        DECLARE_SOFTMAX_KERNEL(neon_qasymm8_softmax);

        DECLARE_SOFTMAX_KERNEL(neon_qasymm8_signed_softmax);

#undef DECLARE_SOFTMAX_KERNEL
    }
}