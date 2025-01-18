//
// Created by Mason on 2025/1/18.
//

#pragma once

#include <data/core/bi_helpers.hpp>

namespace BatmanInfer {
    namespace cpu {
        template<bool IS_LOG>
        void neon_fp32_softmax(const BIITensor *in,
                               void *const tmp,
                               BIITensor *out,
                               const float beta,
                               int axis,
                               const BIWindow &window,
                               const void *lut_ptr);
    }
}
