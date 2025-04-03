//
// Created by Mason on 2025/4/3.
//
#include <cpu/kernels/select/generic/neon/fp32.hpp>
#include <cpu/kernels/select/generic/neon/impl.hpp>
#include <arm_neon.h>

namespace BatmanInfer::cpu {
    void neon_f32_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_32<float, uint32x4_t>(c, x, y, output, window);
    }

    void neon_f32_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_not_same_rank<float>(c, x, y, output, window);
    }
}
