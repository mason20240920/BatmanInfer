//
// Created by Mason on 2025/4/3.
//
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include <cpu/kernels/select/generic/neon/fp16.hpp>
#include <cpu/kernels/select/generic/neon/impl.hpp>
#include <arm_neon.h>
#include <data/core/neon/wrapper/wrapper.hpp>

namespace BatmanInfer::cpu {
    void neon_f16_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_16<float16_t, uint16x8_t>(c, x, y, output, window);
    }

    void neon_f16_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_not_same_rank<float16_t>(c, x, y, output, window);
    }
}

#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)  && defined(ENABLE_FP16_KERNELS) */
