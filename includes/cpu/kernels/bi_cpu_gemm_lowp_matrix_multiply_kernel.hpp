//
// Created by Mason on 2025/1/20.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            class BICpuGemmLowpMatrixMultiplyKernel : public BIICpuKernel<BICpuGemmLowpMatrixMultiplyKernel> {
            public:
                BICpuGemmLowpMatrixMultiplyKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmLowpMatrixMultiplyKernel);

                /**
                 * 初始化内核的输入和输出
                 *
                 * 输入的 @p src0 和 @p src1 都必须是@ref BICpuGemmInterleave4x4Kernel and @ref BICpuGemmTranspose1xWKernel
                 * 的内核. 因为这两个内核会把原始矩阵更 缓存友好
                 *
                 * @param src0 Input tensor info containing the interleaved Matrix A. Data type supported: U8/QASYMM8/S8/QASYMM8_SIGNED
                 * @param src1 Input tensor info containing the transposed1xW Matrix B. Data type supported: U8/QASYMM8/S8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
                 * @param dst Output tensor info to store the result of matrix multiplication. Data type supported: S32
                 */
                void configure(const BIITensorInfo *src0,
                               const BIITensorInfo *src1,
                               BIITensorInfo *dst);

                static BIStatus validate(const BIITensorInfo *src0,
                                         const BIITensorInfo *src1,
                                         const BIITensorInfo *dst);

                void run_op(BatmanInfer::BIITensorPack &tensors, const BatmanInfer::BIWindow &window,
                            const BatmanInfer::ThreadInfo &info) override;

                [[nodiscard]] const char *name() const override;

            private:
                bool _slide_matrix_b{true};
            };
        }
    }
}