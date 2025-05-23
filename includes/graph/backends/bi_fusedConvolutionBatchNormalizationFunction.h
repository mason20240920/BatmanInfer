//
// Created by holynova on 2025/1/15.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_memory_manager.hpp"

namespace BatmanInfer {

namespace graph {

namespace backends {

    /** Wrapper function to first apply BatchNormalizationLayer on the weights and then run ConvolutionLayer with the modified weights */
    template <typename TargetInfo, typename FusedLayerTypes>
    class FusedConvolutionBatchNormalizationFunction : public BIIFunction
    {
    public:
        using TensorType         = typename TargetInfo::TensorType;
        using TensorConcreteType = typename TargetInfo::TensorConcreteType;

        FusedConvolutionBatchNormalizationFunction(std::shared_ptr<BIIMemoryManager> memory_manager = nullptr)
            : _conv_layer(memory_manager), _fused_batch_norm_layer(), _fused_bias(), _is_prepared(false)
        {
        }

        /** Set the input and output tensors.
         *
         * @param[in]  input      Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
         *                        while every optional dimension from 4 and above represent a batch of inputs.
         *                        Data types supported: QASYMM8/F16/F32.
         * @param[in]  weights    Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
         * @param[in]  bias       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
         *                        Data type supported: Should match @p input data type.
         * @param[out] output     Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
         *                        Data types supported: Same as @p input.
         * @param[in]  mean       Mean values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
         * @param[in]  var        Variance values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
         * @param[in]  beta       Beta values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for beta is 0. Data types supported: Same as @p input
         * @param[in]  gamma      Gamma values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for gamma is 1. Data types supported: Same as @p input
         * @param[in]  epsilon    Small value to avoid division with zero. Default value is 0.001f.
         * @param[in]  conv_info  Contains padding and stride information described in @ref PadStrideInfo.
         * @param[in]  num_groups Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
         * @param[in]  fast_math  Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
         *                        available which may introduce a drop of accuracy as well. Default is false
         * @param[in]  fused_act  Activation layer information in case of a fused activation.
         *
         */
        void configure(TensorType                  *input,
                       TensorType                  *weights,
                       TensorType                  *bias,
                       TensorType                  *output,
                       const TensorType            *mean,
                       const TensorType            *var,
                       const TensorType            *beta,
                       const TensorType            *gamma,
                       float                        epsilon,
                       const BIPadStrideInfo       &conv_info,
                       unsigned int                 num_groups,
                       bool                         fast_math,
                       BIActivationLayerInfo const &fused_act)
        {
            // We don't run any validate, as we assume that the layers have been already validated
            const bool        has_bias = (bias != nullptr);
            const TensorType *bias_to_use;

            // We check if the layer has a bias. If yes, use it in-place. If not, we need to create one
            // as batch normalization might end up with a bias != 0
            if (has_bias)
            {
                _fused_batch_norm_layer.configure(weights, mean, var, nullptr, nullptr, bias, beta, gamma, epsilon);
                bias_to_use = bias;
            }
            else
            {
                _fused_batch_norm_layer.configure(weights, mean, var, nullptr, &_fused_bias, nullptr, beta, gamma, epsilon);
                bias_to_use = &_fused_bias;
            }

            _conv_layer.configure(input, weights, bias_to_use, output, conv_info, BIWeightsInfo(), Size2D(1U, 1U), fused_act,
                                  fast_math, num_groups);

            if (!has_bias)
            {
                _fused_bias.allocator()->allocate();
            }
        }

        // Inherited methods overridden:
        void run()
        {
            prepare();
            _conv_layer.run();
        }

        void prepare()
        {
            if (!_is_prepared)
            {
                _fused_batch_norm_layer.run();
                _is_prepared = true;
            }
        }

    private:
        typename FusedLayerTypes::ConvolutionLayer       _conv_layer;
        typename FusedLayerTypes::FuseBatchNormalization _fused_batch_norm_layer;
        TensorConcreteType                               _fused_bias;
        bool                                             _is_prepared;
    };

} // namespace backends

} // namespace graph

} // namespace BatmanInfer
