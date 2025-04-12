//
// Created by Mason on 2025/2/8.
//

#pragma once

#include <cpu/bi_i_cpu_operator.hpp>

namespace BatmanInfer {
    class BIITensorInfo;

    namespace cpu {
        /** Basic function to run @ref kernels::BICpuDequantizeKernel that dequantizes an input tensor */
        class BICpuDequantize : public BIICpuOperator {
        public:
            /** Configure the kernel.
             *
             * @param[in]  src Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/QSYMM8/QSYMM16.
             * @param[out] dst Destination tensor info with the same dimensions of input. Data type supported: F16/F32.
             */
            void configure(const BIITensorInfo *src, BIITensorInfo *dst);

            void dynamic_configure(const BIITensorInfo *src) const;

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref CpuDequantize::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *src, const BIITensorInfo *dst);

            // Inherited methods overridden:
            void run(BIITensorPack &tensors) override;
        };
    } // namespace cpu
}
