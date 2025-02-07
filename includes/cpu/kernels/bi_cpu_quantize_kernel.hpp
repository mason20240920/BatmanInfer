//
// Created by Mason on 2025/2/7.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /** Interface for the quantization layer kernel.
             *
             * @note The implementation supports only 3D input tensors
             */
            class BICpuQuantizeKernel : public BIICpuKernel<BICpuQuantizeKernel> {
            public:
                BICpuQuantizeKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuQuantizeKernel);

                /** Set the input, output.
                 *
                 * @param[in]  src Source tensor info. The dimensions over the third will be interpreted as batches. Data types supported: QASYMM8/QASYMM8_SIGNED/F32/F16.
                 * @param[out] dst Destination tensor info with the same dimensions of input. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16.
                 *
                 * @note Output auto initialization is not supported by this kernel
                 */
                void configure(const BIITensorInfo *src, BIITensorInfo *dst);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to @ref CpuQuantizeKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus validate(const BIITensorInfo *src, const BIITensorInfo *dst);

                /** Get the preferred dimension in which the scheduler splits the work into multiple jobs.
                *
                * @return The split dimension hint.
                */
                size_t get_split_dimension_hint() const {
                    return _split_dimension;
                }

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

            private:
                /** Common signature for all the specialised @ref CpuQuantizeKernel functions
                 *
                 * @param[in] window Region on which to execute the kernel.
                 */
                using QuantizeFunctionExecutorPtr = void (*)(const BIITensor *src, BIITensor *dst,
                                                             const BIWindow &window);
                QuantizeFunctionExecutorPtr _func{nullptr};
                size_t _split_dimension{BIWindow::DimY};
            };
        } // namespace kernels
    }
}