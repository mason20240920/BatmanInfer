//
// Created by Mason on 2025/1/8.
//

#include "cpu/kernels/gemm_matrix_add/generic/neon/fp32.hpp"
#include "cpu/kernels/gemm_matrix_add/generic/neon/impl.hpp"

namespace BatmanInfer {
    namespace cpu {
        void neon_fp32_gemm_matrix_add(const BIITensor *src,
                                       BIITensor *dst,
                                       const BIWindow &window,
                                       float beta) {
            return matrix_addition_f32(src, dst, window, beta);
        }
    }
}