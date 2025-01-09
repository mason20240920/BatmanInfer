//
// Created by Mason on 2025/1/8.
//

#ifndef BATMANINFER_KERNELS_GEMMMA_MATRIX_ADD_LIST_HPP
#define BATMANINFER_KERNELS_GEMMMA_MATRIX_ADD_LIST_HPP

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_window.hpp>
#include <cpu/kernels/gemm_matrix_add/generic/neon/fp32.hpp>
#include <cpu/kernels/gemm_matrix_add/generic/neon/fp16.hpp>

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_GEMMMATRIXADD_KERNEL(func_name) \
    void func_name(const BIITensor *src, BIITensor *dst, const BIWindow &window, float beta)

        DECLARE_GEMMMATRIXADD_KERNEL(neon_fp32_gemm_matrix_add);

        DECLARE_GEMMMATRIXADD_KERNEL(neon_fp16_gemm_matrix_add);

#undef DECLARE_GEMMMATRIXADD_KERNEL
    }
}

#endif //BATMANINFER_KERNELS_GEMMMA_MATRIX_ADD_LIST_HPP
