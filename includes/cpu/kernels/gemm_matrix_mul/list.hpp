//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_MATRIX_MUL_LIST_HPP
#define BATMANINFER_MATRIX_MUL_LIST_HPP

#include <cpu/kernels/gemm_matrix_mul/generic/neon/f32.hpp>
#include <cpu/kernels/gemm_matrix_mul/generic/neon/fp16.hpp>

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_GEMM_MATRIX_MUL_KERNEL(func_name)                                                                        \
    void func_name(const BIITensor *lhs, const BIITensor *rhs, BIITensor *dst, const BIWindow &window, const ThreadInfo &info, \
                   float alpha, const bool is_dst_vector)

        DECLARE_GEMM_MATRIX_MUL_KERNEL(neon_fp32_gemm_matrix_mul);

        DECLARE_GEMM_MATRIX_MUL_KERNEL(neon_fp16_gemm_matrix_mul);

#undef DECLARE_GEMMMATRIXMUL_KERNEL
    }
}

#endif //BATMANINFER_MATRIX_MUL_LIST_HPP
