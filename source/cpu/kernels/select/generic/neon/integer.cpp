//
// Created by Mason on 2025/4/3.
//

#include <cpu/kernels/select/generic/neon/integer.hpp>
#include <cpu/kernels/select/generic/neon/impl.hpp>
#include <arm_neon.h>

namespace BatmanInfer::cpu {
    void neon_s8_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_8<int8_t, uint8x16_t>(c, x, y, output, window);
    }

    void neon_s16_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_16<int16_t, uint16x8_t>(c, x, y, output, window);
    }

    void neon_s32_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_32<int32_t, uint32x4_t>(c, x, y, output, window);
    }

    void neon_s8_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_not_same_rank<int8_t>(c, x, y, output, window);
    }

    void neon_s16_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_not_same_rank<int16_t>(c, x, y, output, window);
    }

    void neon_s32_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_not_same_rank<int32_t>(c, x, y, output, window);
    }

    void neon_u8_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_8<uint8_t, uint8x16_t>(c, x, y, output, window);
    }

    void neon_u16_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_16<uint16_t, uint16x8_t>(c, x, y, output, window);
    }

    void neon_u32_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_32<uint32_t, uint32x4_t>(c, x, y, output, window);
    }

    void neon_u8_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_not_same_rank<uint8_t>(c, x, y, output, window);
    }

    void neon_u16_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_not_same_rank<uint16_t>(c, x, y, output, window);
    }

    void neon_u32_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window) {
        return select_op_not_same_rank<uint32_t>(c, x, y, output, window);
    }
}
