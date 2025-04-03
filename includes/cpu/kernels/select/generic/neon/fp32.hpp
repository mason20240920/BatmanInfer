//
// Created by Mason on 2025/4/3.
//

#pragma once

#include <data/core/bi_tensor_info.hpp>

namespace BatmanInfer::cpu {
    void neon_f32_select_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);

    void neon_f32_select_not_same_rank(
        const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window);
}
