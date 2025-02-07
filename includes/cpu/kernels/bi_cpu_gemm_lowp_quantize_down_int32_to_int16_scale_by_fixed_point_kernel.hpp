//
// Created by Mason on 2025/2/7.
//

#pragma once

#include <data/core/kernel_descriptors.hpp>

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    // Forward declaration
    class BIITensor;
    namespace cpu {
        namespace kernels {
            /** Kernel used to quantize down the int32 accumulator values of GEMMLowp to QSYMM16
             *
             * This kernel takes a final int32 accumulator value (the output of @ref CpuGemmLowpMatrixMultiplyKernel), and processes it to obtain the final QSYMM16 value.
             * The following computations will be performed by the kernel:
             *
             *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
             *  -# Add bias to final result if bias tensor is not a nullptr
             *  -# Round to nearest division by a power-of-two using result_shift
             *  -# Clamp the value between the specified min and max bounds
             *  -# Clamp the resulting int32 values to the [-32768, 32767] range and cast to QSYMM16.
             *
             */
            class BICpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel
                    : public BIICpuKernel<BICpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel> {
            public:
                BICpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel);

                /** Initialise the kernel's input and output.
                 *
                 * @param[in]  src                          Input tensor info. Data type supported: S32
                 * @param[in]  bias                         Biases tensor info. Only shared biases supported and it can be a nullptr if the biases addition is not required.
                 *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
                 * @param[out] dst                          Output tensor info. Data type supported: Data type supported: QSYMM16
                 * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
                 * @param[in]  result_shift                 Integer value used to round to nearest division by a power-of-two the result after the fixed point multiplication
                 * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QSYMM16. Defaults to 0.
                 * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QSYMM16.
                 *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to 0.
                 */
                void configure(BIITensorInfo *src,
                               BIITensorInfo *bias,
                               BIITensorInfo *dst,
                               int result_fixedpoint_multiplier,
                               int result_shift,
                               int min = 0,
                               int max = 0);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to CpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus
                validate(const BIITensorInfo *src, const BIITensorInfo *bias, const BIITensorInfo *dst, int min = 0,
                         int max = 0);

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

            private:
                /** Template function to run the CpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel
                 *
                 * @param[in]  src    Input tensor info
                 * @param[in]  bias   Bias tensor info
                 * @param[out] dst    Output tensor info
                 * @param[in]  window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
                 */
                template<bool is_bounded_relu>
                void run_internal(const BIITensor *src, const BIITensor *bias, BIITensor *dst, const BIWindow &window);

                /** Common signature for all the specialised CpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel functions
                 *
                 * @param[in]  src    Input tensor info
                 * @param[in]  bias   Bias tensor info
                 * @param[out] dst    Output tensor info
                 * @param[in]  window Region on which to execute the kernel.
                 */
                using QuantizeDownFunctionPtr = void (BICpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::*)(
                        const BIITensor *src, const BIITensor *bias, BIITensor *dst, const BIWindow &window);

                QuantizeDownFunctionPtr _func{nullptr};
                int _result_fixedpoint_multiplier{0};
                int _result_shift{0};
                int _min{0};
                int _max{0};
            };
        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer
