//
// Created by Mason on 2025/1/9.
//

#include "cpu/kernels/gemm_matrix_mul/generic/neon/f32.hpp"
#include "cpu/kernels/gemm_matrix_mul/generic/neon/mul.hpp"

namespace BatmanInfer {
    namespace cpu {
        void neon_fp32_gemm_matrix_mul(const BIITensor *lhs,
                                       const BIITensor *rhs,
                                       BIITensor *dst,
                                       const BIWindow &window,
                                       const ThreadInfo &info,
                                       float alpha,
                                       const bool is_dst_vector) {
            return (is_dst_vector) ? vector_matrix_multiply_f32(lhs, rhs, dst, window, info, alpha)
                                   : matrix_matrix_multiply_f32(lhs, rhs, dst, window, info, alpha);
        }
    }
}