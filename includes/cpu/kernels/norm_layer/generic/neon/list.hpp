//
// Created by Mason on 2025/2/9.
//

#pragma once

#include <cpu/kernels/norm_layer/generic/neon/fp16.hpp>
#include <cpu/kernels/norm_layer/generic/neon/fp32.hpp>

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_NORMALIZATION_KERNEL(func_name)                                                      \
    void func_name(const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out, \
                   BINormalizationLayerInfo ninfo)

        DECLARE_NORMALIZATION_KERNEL(neon_normalize_float32_4_0_2D);

        DECLARE_NORMALIZATION_KERNEL(neon_normalize_float32_4_0);

        DECLARE_NORMALIZATION_KERNEL(neon_normalize_float32_4_1_2D);

        DECLARE_NORMALIZATION_KERNEL(neon_normalize_float32_4_1);

        DECLARE_NORMALIZATION_KERNEL(neon_normalize_float32_4_2);

        DECLARE_NORMALIZATION_KERNEL(neon_normalize_float16_8_0_2D);

        DECLARE_NORMALIZATION_KERNEL(neon_normalize_float16_8_0);

        DECLARE_NORMALIZATION_KERNEL(neon_normalize_float16_8_1_2D);

        DECLARE_NORMALIZATION_KERNEL(neon_normalize_float16_8_1);

        DECLARE_NORMALIZATION_KERNEL(neon_normalize_float16_8_2);

#undef DECLARE_NORMALIZATION_KERNEL
    }
} // namespace BatmanInfer