//
// Created by Mason on 2025/8/2.
//

#pragma once

#include <cpu/kernels/layer_norm/generic/neon/fp16.hpp>


namespace BatmanInfer::cpu {
#define DECLARE_LAYER_NORMALIZATION_KERNEL(func_name)                                                      \
void func_name(const BIWindow *window, const BIITensor *input, BIITensor *output, const BIITensor *gamma, const BIITensor *beta, float epsilon)

    DECLARE_LAYER_NORMALIZATION_KERNEL(neon_layer_norm_float16);

#undef DECLARE_LAYER_NORMALIZATION_KERNEL
}
