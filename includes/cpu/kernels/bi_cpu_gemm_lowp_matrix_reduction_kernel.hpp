//
// Created by Mason on 2025/1/20.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    // 前向声明
    struct GEMMLowpReductionKernelInfo;

    namespace cpu {
        namespace kernels {
            /**
             * 用于计算矩阵 A 每行所有元素之和的行向量的内核。
             *  @note This stage is needed to handle the offset of matrix product
             *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
             */
            class BICpuGemmLowpMatrixAReductionKernel : public BIICpuKernel<BICpuGemmLowpMatrixAReductionKernel> {
            public:
                BICpuGemmLowpMatrixAReductionKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmLowpMatrixAReductionKernel);

                /**
                 * 初始化内核的输入和输出
                 * @param src Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
                 * @param dst Output row-vector of sums of all the entries in each row of mtx_a. Data type supported: S32
                 * @param info
                 */
                void configure(const BIITensorInfo *src,
                               BIITensorInfo *dst,
                               const GEMMLowpReductionKernelInfo &info);

                /**
                 * 验证合法性
                 * @param src
                 * @param dst
                 * @param info
                 * @return
                 */
                static BIStatus validate(const BIITensorInfo *src,
                                         const BIITensorInfo *dst,
                                         const GEMMLowpReductionKernelInfo &info);

                void run_op(BatmanInfer::BIITensorPack &tensors,
                            const BatmanInfer::BIWindow &window,
                            const BatmanInfer::ThreadInfo &info) override;

                const char *name() const override;

            private:
                /**
                 * 针对输入类型的缩减内核执行
                 * @tparam T
                 * @param src
                 * @param dst
                 * @param window
                 */
                template<typename T>
                void run_internal(const BIITensor *src,
                                  BIITensor *dst,
                                  const BIWindow &window);

                using BICpuGemmLowpMatrixAReductionKernelPtr = void (BICpuGemmLowpMatrixAReductionKernel::*)(
                    const BIITensor *src,
                    BIITensor *dst,
                    const BIWindow &window);

                BICpuGemmLowpMatrixAReductionKernelPtr _func{nullptr};
                int32_t _k{0};
                int32_t _scalar{0};
                bool _mul_by_scalar{false};
            };

            class BICpuGemmLowpMatrixBReductionKernel : public BIICpuKernel<BICpuGemmLowpMatrixBReductionKernel> {
            public:
                BICpuGemmLowpMatrixBReductionKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmLowpMatrixBReductionKernel);

                /**
                 * 初始化内核的输入和输出
                 * @param src Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
                 * @param dst Output row-vector of sums of all the entries in each row of mtx_b. Data type supported: S32
                 * @param info
                 */
                void configure(const BIITensorInfo *src,
                               BIITensorInfo *dst,
                               const GEMMLowpReductionKernelInfo &info);

                /**
                 * 验证合法性
                 * @param src
                 * @param dst
                 * @param info
                 * @return
                 */
                static BIStatus validate(const BIITensorInfo *src,
                                         const BIITensorInfo *dst,
                                         const GEMMLowpReductionKernelInfo &info);

                void run_op(BatmanInfer::BIITensorPack &tensors,
                            const BatmanInfer::BIWindow &window,
                            const BatmanInfer::ThreadInfo &info) override;

                const char *name() const override;

            private:
                /**
                 * 针对输入类型的缩减内核执行
                 * @tparam T
                 * @param src
                 * @param dst
                 * @param window
                 */
                template<typename T>
                void run_internal(const BIITensor *src,
                                  BIITensor *dst,
                                  const BIWindow &window,
                                  const ThreadInfo &info);

                using BICpuGemmLowpMatrixBReductionKernelPtr = void (BICpuGemmLowpMatrixBReductionKernel::*)(
                    const BIITensor *src,
                    BIITensor *dst,
                    const BIWindow &window,
                    const ThreadInfo &info);

                BICpuGemmLowpMatrixBReductionKernelPtr _func{nullptr};
                int32_t _k{0};
                int32_t _scalar{0};
                bool _mul_by_scalar{false};
            };
        }
    }
}
