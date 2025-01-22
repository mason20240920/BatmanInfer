//
// Created by Mason on 2025/1/22.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /**
             * 转换2D全连接层权重从 NCHW 到 NHWC
             *
             * @note This function can be applied to the 2D weights used by a Fully Connected layer if:
             *       - It follows a Convolution layer
             *       - The data layout used by the network does not match the one the model has been trained in.
             *
             * @note This function assumes the weights are already reshaped (transposed)
             *
             */
            class BICpuConvertFullyConnectedWeightsKernel
                    : public BIICpuKernel<BICpuConvertFullyConnectedWeightsKernel> {
            public:
                BICpuConvertFullyConnectedWeightsKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuConvertFullyConnectedWeightsKernel);

                /**
                 *
                 * @param src Source weights tensor info to convert. Must be 2 dimensional. Data types supported: All.
                 * @param dst The converted weights tensor info. Shape and Data Type: Same as @p src.
                 * @param original_input_shape Shape of the original src tensor (the one entering fully connected layer).
                 * @param data_layout
                 */
                void configure(const BIITensorInfo *src,
                               BIITensorInfo *dst,
                               const BITensorShape &original_input_shape,
                               BIDataLayout data_layout);

                static BIStatus validate(const BIITensorInfo *src,
                                         const BIITensorInfo *dst,
                                         const BITensorShape &original_input_shape,
                                         BIDataLayout data_layout);

                void run_op(BatmanInfer::BIITensorPack &tensors,
                            const BatmanInfer::BIWindow &window,
                            const BatmanInfer::ThreadInfo &info) override;

                const char *name() const override;

            private:
                /* 如果@p data_layout == NCHW，则等于原始src平面中每个元素的数量；否则，等于其通道数 */
                unsigned int _factor1{
                        0};
                /* 如果@p data_layout == NHWC，则等于原始src平面中每个元素的数量；否则，等于其通道数 */
                unsigned int _factor2{
                        0};
            };
        }
    }
}