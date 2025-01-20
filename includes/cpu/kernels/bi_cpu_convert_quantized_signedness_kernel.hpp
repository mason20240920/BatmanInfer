//
// Created by Mason on 2025/1/20.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /**
             * Kernel: 转换非对称有符号到非对称有符号?
             */
            class BICpuConvertQuantizedSignednessKernel : public BIICpuKernel<BICpuConvertQuantizedSignednessKernel> {
            public:
                BICpuConvertQuantizedSignednessKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuConvertQuantizedSignednessKernel);

                void configure(const BIITensorInfo *src,
                               BIITensorInfo *dst);

                static BIStatus validate(const BIITensorInfo *src,
                                         const BIITensorInfo *dst);

                void run_op(BatmanInfer::BIITensorPack &tensors,
                            const BatmanInfer::BIWindow &window,
                            const BatmanInfer::ThreadInfo &info) override;

                const char *name() const override;
            };
        } // namespace kernels
    }  // namespace cpu
}