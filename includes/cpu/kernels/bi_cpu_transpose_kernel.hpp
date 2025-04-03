//
// Created by Mason on 2025/1/6.
//

#ifndef BATMANINFER_BI_CPU_TRANSPOSE_KERNEL_HPP
#define BATMANINFER_BI_CPU_TRANSPOSE_KERNEL_HPP

#include "data/core/common/bi_core_common_macros.hpp"
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /**
             * @brief 转置矩阵元素的内核
             */
            class BICpuTransposeKernel : public BIICpuKernel<BICpuTransposeKernel> {
            public:
                BICpuTransposeKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuTransposeKernel);

                /**
                 * @brief 为给定的参数列表配置内核
                 * @param src 源张量进行置换。支持的数据类型：全部。
                 * @param dst 目标张量, 支持数据类型: @p src 相同
                 */
                void configure(const BIITensorInfo *src,
                               BIITensorInfo *dst);

                /**
                 * @brief 动态参数进行修改
                 * @param src
                 * @param dst
                 */
                void dynamic_configure(const BIITensorInfo *src,
                                       BIITensorInfo *dst);

                /**
                 * @brief 确定是否配置有效
                 * @param src
                 * @param dst
                 * @return
                 */
                static BIStatus validate(const BIITensorInfo *src,
                                         const BIITensorInfo *dst);

                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;
            };
        }
    }
}

#endif //BATMANINFER_BI_CPU_TRANSPOSE_KERNEL_HPP
