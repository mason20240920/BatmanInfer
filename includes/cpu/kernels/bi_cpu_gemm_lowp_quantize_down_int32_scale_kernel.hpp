//
// Created by Mason on 2025/2/7.
//

#pragma once

#include <data/core/kernel_descriptors.hpp>

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    namespace cpu {
        namespace kernels {
            /** Kernel used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8/QASYMM8_SIGNED
             *
             * This kernel takes a final int32 accumulator value (the output of @ref CpuGemmLowpMatrixMultiplyKernel), and processes it to obtain the final QASYMM8/QASYMM8_SIGNED value.
             * The following computations will be performed by the kernel:
             *
             *  -# Add offset terms to final result
             *  -# Multiply each entry of result by result_mult_int
             *  -# Add bias to final result if bias tensor is not a nullptr
             *  -# Shift the int32 accumulator by result_shift
             *  -# Clamp the value between the specified min and max bounds
             *  -# Clamp the resulting int32 values:
             *  -#  -to the [0..255] range and cast to QASYMM8.
             *  -#  -to the [-128..127] range and cast to QASYMM8_SIGNED.
             *
             */
            class BICpuGemmLowpQuantizeDownInt32ScaleKernel
                    : public BIICpuKernel<BICpuGemmLowpQuantizeDownInt32ScaleKernel> {
            public:
                BICpuGemmLowpQuantizeDownInt32ScaleKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmLowpQuantizeDownInt32ScaleKernel);

                /** Initialise the kernel's input and output.
                 *
                 * @param[in]  src          Input tensor info. Data type supported: S32
                 * @param[in]  bias         Biases tensor info. Only shared biases supported and it can be a nullptr if the biases addition is not required.
                 *                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
                 * @param[out] dst          Output tensor info. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED
                 * @param[out] output_stage GEMMLowp output stage metadata.
                 */
                void configure(BIITensorInfo *src, BIITensorInfo *bias, BIITensorInfo *dst,
                               const BIGEMMLowpOutputStageInfo *output_stage);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to CpuGemmLowpQuantizeDownInt32ScaleKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus validate(const BIITensorInfo *src,
                                         const BIITensorInfo *bias,
                                         const BIITensorInfo *dst,
                                         const BIGEMMLowpOutputStageInfo *output_stage);

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

            private:
                /** Template function to run the NEGEMMLowpQuantizeDownInt32ScaleKernel
                 *
                 * @param[in]  src    Input tensor info
                 * @param[in]  bias   Biases tensor info
                 * @param[out] dst    Output tensor info
                 * @param[in]  window Region on which to execute the kernel. (Must be a valid region of the window returned by window())
                 */
                template<typename T>
                void run_internal(const BIITensor *src, const BIITensor *bias, BIITensor *dst, const BIWindow &window);

                /** Common signature for all the specialised CpuGemmLowpQuantizeDownInt32ScaleKernel functions
                 *
                 * @param[in]  src    Input tensor info
                 * @param[in]  bias   Biases tensor info
                 * @param[out] dst    Output tensor info
                 * @param[in]  window Region on which to execute the kernel.
                 */
                using QuantizeDownFunctionPtr = void (BICpuGemmLowpQuantizeDownInt32ScaleKernel::*)(
                        const BIITensor *src,
                        const BIITensor *bias,
                        BIITensor *dst,
                        const BIWindow &window);

                QuantizeDownFunctionPtr _func{nullptr};
                const BIGEMMLowpOutputStageInfo *_output_stage{nullptr};
                bool _is_bounded_relu{false};
            };
        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer