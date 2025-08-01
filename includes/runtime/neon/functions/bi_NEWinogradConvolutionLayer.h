//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "function_info/bi_activationLayerInfo.h"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_memory_manager_on_demand.hpp"
#include "runtime/bi_tensor.hpp"

#include <memory>

namespace BatmanInfer {

    // Forward declarations
    class BIITensor;

    /** Basic function to simulate a convolution layer. This function calls the following kernels:
     *
     * -# cpu::CpuWinogradConv2dTransformInputKernel
     * -# cpu::CpuWinogradConv2dTransformOutputKernel
     * -# cpu::CpuGemmAssemblyDispatch
     * -# @ref CPPPermute (three times: weights, input and output)
     *
     * @note  Some Winograd configurations (i.e. F(2x2, 5x5), F(4x4, 5x5)) are supported only with enable_fast_math = true
     */
    class BINEWinogradConvolutionLayer : public BIIFunction
    {
    public:
        /** Constructor */
        BINEWinogradConvolutionLayer(const std::shared_ptr<BIIMemoryManager> &memory_manager);
        BINEWinogradConvolutionLayer() : BINEWinogradConvolutionLayer(BIMemoryManagerOnDemand::make_default())
        {
        }
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEWinogradConvolutionLayer(const BINEWinogradConvolutionLayer &) = delete;
        /** Default move constructor */
        BINEWinogradConvolutionLayer(BINEWinogradConvolutionLayer &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEWinogradConvolutionLayer &operator=(const BINEWinogradConvolutionLayer &) = delete;
        /** Default move assignment operator */
        BINEWinogradConvolutionLayer &operator=(BINEWinogradConvolutionLayer &&) = default;
        /** Destructor */
        ~BINEWinogradConvolutionLayer();

        /** Set the input and output tensors.
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src0           |src1           |src2   |dst            |
         * |:--------------|:--------------|:------|:--------------|
         * |F16            |F16            |F16    |F16            |
         * |F32            |F32            |F32    |F32            |
         *
         * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
         *                              while every optional dimension from 4 and above represent a batch of inputs.
         *                              Data types supported: F16/F32.
         * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
         *                              Supported kernel sizes: (height, width) -> 3x3, 1x3, 3x1, 5x5, 1x5, 5x1 for Fp32
         *                              -> 3x3 for Fp16
         * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
         * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
         *                              Data types supported: Same as @p input.
         * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo. Currently only unit strides are supported.
         * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
         * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
         *                              available which may introduce a drop of accuracy as well. Default is false
         */
        void configure(const BIITensor             *input,
                       const BIITensor             *weights,
                       const BIITensor             *biases,
                       BIITensor                   *output,
                       const BIPadStrideInfo       &conv_info,
                       const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                       bool                         enable_fast_math = false);

        // Inherited methods overridden:
        void run();
        void prepare() override;

        /** Static function to check if given info will lead to a valid configuration of @ref NEWinogradConvolutionLayer
         *
         * Similar to @ref NEWinogradConvolutionLayer::configure()
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo         *input,
                                 const BIITensorInfo         *weights,
                                 const BIITensorInfo         *biases,
                                 const BIITensorInfo         *output,
                                 const BIPadStrideInfo       &conv_info,
                                 const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                                 bool                         enable_fast_math = false);

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
