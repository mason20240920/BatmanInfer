//
// Created by Mason on 2025/1/8.
//

#ifndef BATMANINFER_BI_CPU_GEMM_INTER_LEAVE_4X4_KERNEL_HPP
#define BATMANINFER_BI_CPU_GEMM_INTER_LEAVE_4X4_KERNEL_HPP

#include "data/core/common/bi_core_common_macros.hpp"
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /**
             * 用于交错矩阵元素的内核
             *
             * 此函数将矩阵 A 的 4x4 块中的值放置在同一行中（交错后的值）。
             *
             * @f[
             * \left( \begin{array}{cccc}
             * a00 & a01 & a02 & a03 \\
             * a10 & a11 & a12 & a13 \\
             * a20 & a21 & a22 & a23 \\
             * a30 & a31 & a32 & a33 \\
             * \end{array} \right)
             * \rightarrow
             * \left( \begin{array}{ccccccccccccccccc}
             * a00 & a10 & a20 & a30 & a01 & a11 & a21 & a31 & a02 & a12 & a22 & a32 & a03 & a13 & a23 & a33 \\
             * \end{array} \right)
             * @f]
             *
             * 1. 增加连续分块
             * 2. 增加SIMD命中
             * 3. 增加缓存命中
             * 在此操作之后，目标矩阵将具有以下形状：[ height * 4, ceil(width / 4.0f) ]
             */
            class BICpuGemmInterleave4x4Kernel : public BIICpuKernel<BICpuGemmInterleave4x4Kernel> {
            public:
                BICpuGemmInterleave4x4Kernel() = default;

                /**
                 * 初始化内核的源和目标张量信息
                 *
                 * @param src 输入的张量信息。数据类型支持: ALL
                 * @param dst 输出的张量(存储交错信息), 数据类型: 和@p src相同
                 */
                void configure(const BIITensorInfo *src, BIITensorInfo *dst);

                /**
                 * 静态方法验证给定的信息是否是合法的
                 *
                 * @param src
                 * @param dst
                 * @return
                 */
                static BIStatus validate(const BIITensorInfo *src,
                                         const BIITensorInfo *dst);

                void run_op(BIITensorPack &tensors,
                            const BIWindow &window,
                            const ThreadInfo &info) override;

                const char *name() const override;
            };
        }
    }
}

#endif //BATMANINFER_BI_CPU_GEMM_INTER_LEAVE_4X4_KERNEL_HPP
