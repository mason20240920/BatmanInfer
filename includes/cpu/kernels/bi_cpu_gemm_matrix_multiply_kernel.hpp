//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_BI_CPU_GEMM_MATRIX_MULTIPLY_KERNEL_HPP
#define BATMANINFER_BI_CPU_GEMM_MATRIX_MULTIPLY_KERNEL_HPP

#include "data/core/common/bi_core_common_macros.hpp"
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
            public:
                struct BIGemmMatrixMulKernel {
                    const char *name;
                    const DataTypeISASelectorPtr is_selected;
                    BIGemmMatrixMulKernelPtr ukernel;
                };

                BICpuGemmMatrixMultiplyKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmMatrixMultiplyKernel);

                /**
                 * 初始化内核的输入和输出。
                 *
                 * @note 如果输出张量是一个矩阵，则输入矩阵 `lhs` 和 `rhs` 应该是以下内核的输出：`BICpuGemmInterleave4x4Kernel` 和 `BICpuGemmTranspose1xWKernel`。
                 *       这两个内核会改变原始矩阵的布局，使其更符合缓存友好的访问模式。
                 *
                 * @param lhs 左侧张量信息，包含交错后的矩阵 A 或向量 A。支持的数据类型：F16/F32
                 * @param rhs 右侧张量信息，包含转置后的矩阵 B（如果第一个输入张量 A 不是向量）如果输出张量是一个向量，则 `rhs` 必须包含未重排的矩阵 B。支持的数据类型：与 `lhs` 相同。
                 * @param dst 输出张量，用于存储矩阵乘法的结果。支持的数据类型：与 `lhs` 相同
                 * @param alpha 矩阵乘积的权重
                 * @param is_interleaved 如果 `lhs` 和 `rhs` 分别通过 `CpuGemmInterleave4x4Kernel` 和 `CpuGemmTranspose1xWKernel` 进行了重排，则为 true。
                 * @param reshape_info GEMM 重排信息。如果 `is_interleaved_transposed = true`，则此对象必须包含理解 `lhs` 和 `rhs` 如何重排的信息
                 */
                void configure(const BIITensorInfo *lhs,
                               const BIITensorInfo *rhs,
                               BIITensorInfo *dst,
                               float alpha,
                               bool is_interleaved,
                               const BIGemmReshapeInfo &reshape_info = BIGemmReshapeInfo());

                /**
                 * 验证有效性
                 * @param lhs
                 * @param rhs
                 * @param dst
                 * @param alpha
                 * @param is_interleaved
                 * @param reshape_info
                 * @return
                 */
                static BIStatus validate(const BIITensorInfo *lhs,
                                         const BIITensorInfo *rhs,
                                         const BIITensorInfo *dst,
                                         float alpha,
                                         bool is_interleaved,
                                         const BIGemmReshapeInfo &reshape_info);

                void run_op(BIITensorPack &tensors,
                            const BIWindow &window,
                            const ThreadInfo &info) override;

                const char *name() const override;

                static const std::vector<BIGemmMatrixMulKernel> &get_available_kernels();

            private:
                /** Common signature for all the matrix multiply functions
                *
                * @param[in]  lhs    Left-handside input tensor. Data types supported: F16/F32
                * @param[in]  rhs    Right-handside input tensor. Data types supported: same as @p lhs
                * @param[out] dst    The output tensor. Data type supported: same as @p rhs
                * @param[in]  window Region on which to execute the kernel.
                * @param[in]  info   Thread info metadata.
                * @param[in]  alpha  Weight of the matrix product.
                */

                BIGemmMatrixMulKernelPtr _func{nullptr};
                float _alpha{1.f};
            };
        }
    }
}

#endif //BATMANINFER_BI_CPU_GEMM_MATRIX_MULTIPLY_KERNEL_HPP
