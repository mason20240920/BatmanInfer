//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_BI_CPU_GEMM_MATRIX_MULTIPLY_KERNEL_HPP
#define BATMANINFER_BI_CPU_GEMM_MATRIX_MULTIPLY_KERNEL_HPP

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /**
             * 用于将两个输入矩阵 "A" 和 "B" 相乘的内核。矩阵相乘后，输出矩阵/向量的所有元素将乘以 alpha
             *
             * @note 如果输出张量是一个矩阵，则实现假定输入张量 @p lhs 和 @p rhs 均为矩阵，并且分别通过 @ref BICpuGemmInterleave4x4Kernel
             * 和 @ref BICpuGemmTranspose1xWKernel 进行了重排
             *
             * @note 如果输出张量是一个向量且数据类型为 F32，则实现假定第一个输入张量 @p lhs 是一个向量，第二个输入张量 @p rhs 是一个矩阵。实现还假定两个张量均未经过重排
             */
            class BICpuGemmMatrixMultiplyKernel : public BIICpuKernel<BICpuGemmMatrixMultiplyKernel> {
            private:
                using BIGemmMatrixMulKernelPtr = std::add_pointer<void(
                        const BIITensor *,
                        const BIITensor *,
                        BIITensor *,
                        const BIWindow &,
                        const ThreadInfo &,
                        float,
                        const bool)>::type;
            };
        }
    }
}

#endif //BATMANINFER_BI_CPU_GEMM_MATRIX_MULTIPLY_KERNEL_HPP
