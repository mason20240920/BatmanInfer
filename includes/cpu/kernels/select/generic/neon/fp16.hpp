//
// Created by Mason on 2025/4/3.
//

#pragma once
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include <data/core/bi_helpers.hpp>

namespace BatmanInfer::cpu {
    void neon_f16_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_f16_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);
}

#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)  && defined(ENABLE_FP16_KERNELS) */
