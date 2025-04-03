//
// Created by Mason on 2025/4/3.
//

#pragma once
#include <data/core/bi_helpers.hpp>

namespace BatmanInfer::cpu {
    void neon_s8_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_s16_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_s32_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_s8_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_s16_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_s32_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_u8_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_u16_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_u32_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_u8_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_u16_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_u32_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);
}
