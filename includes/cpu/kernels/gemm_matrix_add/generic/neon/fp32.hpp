//
// Created by Mason on 2025/1/8.
//

#ifndef BATMANINFER_MATRIX_ADD_FP32_HPP
#define BATMANINFER_MATRIX_ADD_FP32_HPP

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_window.hpp>

namespace BatmanInfer {
    namespace cpu {
        void neon_fp32_gemm_matrix_add(const BIITensor *src,
                                       BIITensor *dst,
                                       const BIWindow &window,
                                       float beta);
    }
}

#endif //BATMANINFER_MATRIX_ADD_FP32_HPP
