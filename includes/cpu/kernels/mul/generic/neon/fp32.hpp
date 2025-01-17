//
// Created by Mason on 2025/1/17.
//

#pragma once

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_tensor_info.hpp>


namespace BatmanInfer {
    namespace cpu {
        void
        mul_F32_F32_F32(const BIITensor *src1, const BIITensor *src2, BIITensor *out, const BIWindow &window,
                        float scale);
    }
}