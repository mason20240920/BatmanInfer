//
// Created by Mason on 2025/1/16.
//

#pragma once

#include "neon/fp32.hpp"
#include "neon/fp16.hpp"
#include "neon/integer.hpp"
#include "neon/qasymm8.hpp"
#include "neon/qasymm8_signed.hpp"

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_ELEMETWISE_BINARY_KERNEL(func_name) \
    template <ArithmeticOperation op>               \
    void func_name(const BIITensor *in1, const BIITensor *in2, BIITensor *out, const BIWindow &window)

        DECLARE_ELEMETWISE_BINARY_KERNEL(neon_qasymm8_signed_elementwise_binary);

        DECLARE_ELEMETWISE_BINARY_KERNEL(neon_qasymm8_elementwise_binary);

        DECLARE_ELEMETWISE_BINARY_KERNEL(neon_fp16_elementwise_binary);

        DECLARE_ELEMETWISE_BINARY_KERNEL(neon_fp32_elementwise_binary);

        DECLARE_ELEMETWISE_BINARY_KERNEL(neon_s16_elementwise_binary);

        DECLARE_ELEMETWISE_BINARY_KERNEL(neon_s32_elementwise_binary);

#undef DECLARE_ELEMETWISE_BINARY_KERNEL

#define DECLARE_COPMP_ELEMETWISE_BINARY_KERNEL(func_name) \
    template <ComparisonOperation op>                     \
    void func_name(const BIITensor *in1, const BIITensor *in2, BIITensor *out, const BIWindow &window)

        DECLARE_COPMP_ELEMETWISE_BINARY_KERNEL(neon_qasymm8_comparison_elementwise_binary);

        DECLARE_COPMP_ELEMETWISE_BINARY_KERNEL(neon_qasymm8_signed_comparison_elementwise_binary);

        DECLARE_COPMP_ELEMETWISE_BINARY_KERNEL(neon_fp16_comparison_elementwise_binary);

        DECLARE_COPMP_ELEMETWISE_BINARY_KERNEL(neon_u8_comparison_elementwise_binary);

        DECLARE_COPMP_ELEMETWISE_BINARY_KERNEL(neon_s16_comparison_elementwise_binary);

        DECLARE_COPMP_ELEMETWISE_BINARY_KERNEL(neon_s32_comparison_elementwise_binary);

        DECLARE_COPMP_ELEMETWISE_BINARY_KERNEL(neon_fp32_comparison_elementwise_binary);

#undef DECLARE_COPMP_ELEMETWISE_BINARY_KERNEL
    }
}