//
// Created by Mason on 2025/3/27.
//

#pragma once

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_window.hpp>
#include <cpu/kernels/dynamic_gemm/generic/neon/fp32.hpp>

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_DYNAMIC_GEMM_KERNEL(kernel_name)                                                                \
    void   kernel_name##_run(const BIITensor *a, const BIITensor *b, const BIITensor *c, BIITensor *d, BIITensor *pack_b, \
                             const BIWindow &window);                                                             \
    void   kernel_name##_pack_rhs(const BIITensor *rhs, const BIITensor *bias, BIITensor *pack_b);                    \
    size_t kernel_name##_size_of_packed_rhs(size_t rows, size_t columns);                                       \
    BIWindow kernel_name##_window(const BIITensorInfo *dst)

#if defined(__aarch64__) && defined(ENABLE_FP32_KERNELS)

        DECLARE_DYNAMIC_GEMM_KERNEL(neon_fp32_dynamic_gemm);

#endif // __aarch64__ && ENABLE_FP32_KERNELS

#undef DECLARE_DYNAMIC_GEMM_KERNEL
    }
}
