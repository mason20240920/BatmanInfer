//
// Created by Mason on 2025/2/7.
//

#pragma once

#include <cpu/bi_i_cpu_operator.hpp>

namespace BatmanInfer {
    class BIITensorInfo;
    namespace cpu {
        /** Basic function to run @ref kernels::CpuQuantizeKernel that dequantizes an input tensor */
        class BICpuQuantize : public BIICpuOperator {
        public:
            /** Set the input and output tensors.
             *
             * @param[in]  src Source tensor info. The dimensions over the third will be interpreted as batches. Data types supported: QASYMM8/QASYMM8_SIGNED/F32/F16.
             * @param[out] dst Destination tensor info with the same dimensions of input. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16
             */
            void configure(const BIITensorInfo *src, BIITensorInfo *dst);

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref CpuQuantize::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *src, const BIITensorInfo *dst);

            // Inherited methods overridden:
            void run(BIITensorPack &tensors) override;
        };
    } // namespace cpu
} // namespace BatmanInfer