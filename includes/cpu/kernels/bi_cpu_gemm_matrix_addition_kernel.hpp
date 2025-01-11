//
// Created by Mason on 2025/1/8.
//

#ifndef BATMANINFER_BI_CPU_GEMM_MATRIX_ADDITION_KERNEL_HPP
#define BATMANINFER_BI_CPU_GEMM_MATRIX_ADDITION_KERNEL_HPP

#include "data/core/common/bi_core_common_macros.hpp"
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /**
             * 用于执行两个矩阵之间的原地加法的内核，同时考虑第二个矩阵可能会被标量值 beta 加权
             *
             *    @note [ MTX_OUT = MTX_0 + beta * MTX_1 ]，其中 MTX_0 和 MTX_1 的大小相同。
             *    @note 该阶段用于完成 GEMM（广义矩阵乘法）的结果计算，仅在 beta != 0.0 的情况下执行。如果此内核用于完成 GEMM 的结果计算，我们有：

             *   MTX_0 = A * B * alpha，其中 MTX_0 是 @ref biCpuGemmMatrixMultiplyKernel 的输出结果。
             *   MTX_1 = C
             */
            class BICpuGemmMatrixAdditionKernel : public BIICpuKernel<BICpuGemmMatrixAdditionKernel> {
            private:
                using BIGemmMatrixAddKernelPtr = std::add_pointer<void(const BIITensor *,
                                                                       BIITensor *,
                                                                       const BIWindow &,
                                                                       float)>::type;

            public:
                struct BIGemmMatrixAddKernel {
                    const char *name;
                    const DataTypeISASelectorPtr is_selected;
                    BIGemmMatrixAddKernelPtr ukernel;
                };

                BICpuGemmMatrixAdditionKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmMatrixAdditionKernel);

                /**
                 * 初始化内核的输入和输出
                 *
                 * @note 输入和输出张量必须有相同维度
                 *
                 * @param src 输入张量(Matrix C). 数据类型支持： F16/F32
                 * @param dst 输出张量信息。如果此内核用于完成 GEMM（通用矩阵乘法）结果的最终处理，输出将包含由内核 @ref CpuGemmMatrixMultiplyKernel 获得的结果。支持的数据类型：与 @p src 相同。
                 * @param beta 矩阵C的权重
                 */
                void configure(const BIITensorInfo *src,
                               BIITensorInfo *dst,
                               float beta);

                /**
                 * 静态方法: 验证给定的信息是否是有效配置 @ref BICpuGemmMatrixAdditionKernel
                 * @param src
                 * @param dst
                 * @param beta
                 * @return
                 */
                static BIStatus validate(const BIITensorInfo *src,
                                         const BIITensorInfo *dst,
                                         float beta);

                void run_op(BIITensorPack &tensors,
                            const BIWindow &window,
                            const ThreadInfo &info) override;

                const char *name() const override;

                static const std::vector<BIGemmMatrixAddKernel> &get_available_kernels();

            private:
                /**
                 * 所有矩阵加法的方法: 共同签名
                 *
                 *  @param[in]  src    An input tensor. Data types supported: F16/F32
                 *  @param[out] dst    The output tensor. Data type supported: same as @p src
                 *  @param[in]  window Region on which to execute the kernel.
                 *  @param[in]  beta   Weight of matrix C
                 */
                // 用于配置()中传递的特定张量类型的矩阵加法函数
                BIGemmMatrixAddKernelPtr _func{nullptr};
                float _beta{0.f};
            };
        } // namespace kernels
    } // namespace cpu
} //

#endif //BATMANINFER_BI_CPU_GEMM_MATRIX_ADDITION_KERNEL_HPP
