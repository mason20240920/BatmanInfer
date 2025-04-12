//
// Created by Mason on 2025/2/8.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /** Interface for the dequantization layer kernel. */
            class BICpuDequantizeKernel : public BIICpuKernel<BICpuDequantizeKernel> {
            public:
                BICpuDequantizeKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuDequantizeKernel);

                /** Set input, output tensors.
                 *
                 * @param[in]  src Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/QSYMM8/QSYMM16.
                 * @param[out] dst Destination tensor info with the same dimensions of input. Data type supported: F16/F32.
                 */
                void configure(const BIITensorInfo *src, BIITensorInfo *dst);

                void dynamic_configure(const BIITensorInfo *src);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to @ref CpuDequantizeKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus validate(const BIITensorInfo *src, const BIITensorInfo *dst);

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

            private:
                /** Common signature for all the specialised @ref CpuDequantizeKernel functions
                 *
                 * @param[in] window Region on which to execute the kernel.
                 */
                using DequantizeFunctionExecutorPtr = void (*)(const BIITensor *input, BIITensor *output,
                                                               const BIWindow &window);
                DequantizeFunctionExecutorPtr _func{nullptr};
            };
        } // namespace kernels
    }
}
