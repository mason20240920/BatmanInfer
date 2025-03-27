//
// Created by Mason on 2025/3/27.
//

#pragma once

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_steps.hpp>
#include <data/core/bi_window.hpp>

#include <common/utils/bi_validate.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>

namespace BatmanInfer {
    namespace cpu {
#if defined(__aarch64__) && defined(ENABLE_FP32_KERNELS)

        /**
         * 1. 数据重排优化
         * 2. 性能提升目的
         * *  优化内存访问模式，提高缓存命中率
         * *  让数据更好地适配NEON SIMD指令的并行处理能力
         * *  减少内存访问延迟，提升计算效率
         * 3. 特殊处理
         * *  集成了偏置(bias)处理，可能是为了在后续计算中减少额外的操作步骤
         * *  函数针对FP32(单精度浮点数)进行优化，这从函数名和条件编译宏可以看出
         * @param rhs
         * @param bias
         * @param pack_b
         */
        void neon_fp32_dynamic_gemm_pack_rhs(const BIITensor *rhs,
                                             const BIITensor *bias,
                                             BIITensor *pack_b);

        /**
         *
         * @param a
         * @param b
         * @param c
         * @param d
         * @param pack_b
         * @param window
         */
        void neon_fp32_dynamic_gemm_run(
                const BIITensor *a,
                const BIITensor *b,
                const BIITensor *c,
                BIITensor *d,
                BIITensor *pack_b,
                const BIWindow &window);

        size_t neon_fp32_dynamic_gemm_size_of_packed_rhs(size_t rows, size_t columns);

        BIWindow neon_fp32_dynamic_gemm_window(const BIITensorInfo *dst);

#endif // __aarch64__ && ENABLE_FP32_KERNELS
    }
}