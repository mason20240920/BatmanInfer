//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_BI_CPU_GEMM_TRANSPOSE_1XW_KERNEL_HPP
#define BATMANINFER_BI_CPU_GEMM_TRANSPOSE_1XW_KERNEL_HPP

#include "data/core/common/bi_core_common_macros.hpp"
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /**
             * @brief 按 1xW 的块对矩阵元素进行转置的内核。
             *
             * 该内核将输入矩阵的元素重新排列为 1xW 的块形式，其中 W 定义为 `(16 / 张量元素大小)`。
             * 转置后的输出矩阵形状为 `[height * W, ceil(width / W)]`。
             *
             * ### 示例：当输入数据类型为 F32 时
             * 输入矩阵：
             * ```
             * a00  a01  a02  a03
             * a10  a11  a12  a13
             * a20  a21  a22  a23
             * a30  a31  a32  a33
             * ```
             * 转置后的输出矩阵：
             * ```
             * a00  a01  a02  a03  a10  a11  a12  a13  a20  a21  a22  a23  a30  a31  a32  a33
             * ```
             *
             * ### 示例：当输入数据类型为 F16 时
             * 输入矩阵：
             * ```
             * a00  a01  a02  a03  a04  a05  a06  a07
             * a10  a11  a12  a13  a14  a15  a16  a17
             * a20  a21  a22  a23  a24  a25  a26  a27
             * a30  a31  a32  a33  a34  a35  a36  a37
             * ```
             * 转置后的输出矩阵：
             * ```
             * a00  a01  a02  a03  a04  a05  a06  a07  a10  a11  a12  a13  a14  a15  a16  a17 a20  a21  a22  a23  a24  a25  a26  a27  a30  a31  a32  a33  a34  a35  a36  a37
             * ```
             *
             * @note 输出矩阵的形状为 `[height * W, ceil(width / W)]`，其中 `W = 16 / 张量元素大小`。
             *
             * @details
             * 该内核支持所有类型的数据张量。配置方法和验证方法如下：
             *
             * - `configure(const ITensorInfo *src, ITensorInfo *dst)`：
             *   配置内核所需的输入和输出张量信息。
             * - `validate(const ITensorInfo *src, const ITensorInfo *dst)`：
             *   静态函数，用于检查给定的张量信息是否会导致有效的配置。
             */
            class BICpuGemmTranspose1xWKernel : public BIICpuKernel<BICpuGemmTranspose1xWKernel> {
            public:
                BICpuGemmTranspose1xWKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmTranspose1xWKernel);

                /**
                 * 配置运行内核参数
                 * @param src
                 * @param dst
                 */
                void configure(const BIITensorInfo *src,
                               BIITensorInfo *dst);

                /**
                 * 可用配置
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


#endif //BATMANINFER_BI_CPU_GEMM_TRANSPOSE_1XW_KERNEL_HPP
