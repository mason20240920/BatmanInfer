//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "function_info/bi_activationLayerInfo.h"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_memory_manager.hpp"
#include "runtime/bi_memory_group.hpp"

#include <memory>

namespace BatmanInfer {

    class BIITensor;
    class BIITensorInfo;

    /** Function to run the direct convolution.
     *
     *  This function calls the following:
     *
     * -# cpu::CpuDirectConv2d
     */
    class BINEDirectConvolutionLayer : public BIIFunction
    {
    public:
        /** Constructor */
        BINEDirectConvolutionLayer(std::shared_ptr<BIIMemoryManager> memory_manager = nullptr);
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEDirectConvolutionLayer(const BINEDirectConvolutionLayer &) = delete;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEDirectConvolutionLayer &operator=(const BINEDirectConvolutionLayer &) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEDirectConvolutionLayer(BINEDirectConvolutionLayer &&) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEDirectConvolutionLayer &operator=(BINEDirectConvolutionLayer &&) = delete;
        /** Default destructor */
        ~BINEDirectConvolutionLayer();
        /** Set the input, weights, biases and output tensors.
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src0   |src1   |src2   |dst    |
         * |:------|:------|:------|:------|
         * |F16    |F16    |F16    |F16    |
         * |F32    |F32    |F32    |F32    |
         *
         * @note: DirectConvolution only works in the following configurations:
         *    1x1 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F16/F32
         *    3x3 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F16/F32
         *    5x5 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F32
         *
         * @param[in, out] input     Input tensor. Data types supported: F16/F32.
         * @param[in]      weights   Set of kernels to convolve the input volume.
         *                           Supported sizes: 1x1, 3x3 and 5x5.
         *                           The 3rd dimension must be the same as the input's volume 3rd dimension.
         *                           Data type supported: Same as @p input.
         * @param[in]      bias      Set of biases. Can be nullptr. Data type supported: Same as @p input.
         * @param[out]     output    Output tensor.
         *                           The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: Same as @p input.
         * @param[in]      conv_info Contains padding and stride information described in @ref PadStrideInfo.
         * @param[in]      act_info  (Optional) Activation layer information in case of a fused activation.
         */
        void configure(BIITensor                   *input,
                       const BIITensor             *weights,
                       const BIITensor             *bias,
                       BIITensor                   *output,
                       const BIPadStrideInfo       &conv_info,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());
        /** Static function to check if given info will lead to a valid configuration of @ref NEDirectConvolutionLayer
         *
         * @note: DirectConvolution only works in the following configurations:
         *    1x1 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F16/F32
         *    3x3 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F16/F32
         *    5x5 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F32
         *
         * @param[in] input     Input tensor. Data types supported: F16/F32.
         * @param[in] weights   Set of kernels to convolve the input volume.
         *                      Supported sizes: 1x1, 3x3 and 5x5.
         *                      The 3rd dimension must be the same as the input's volume 3rd dimension.
         *                      Data type supported: Same as @p input.
         * @param[in] bias      Set of biases. Can be nullptr. Data type supported: Same as @p input.
         * @param[in] output    Output tensor.
         *                      The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: Same as @p input.
         * @param[in] conv_info Contains padding and stride information described in @ref PadStrideInfo.
         * @param[in] act_info  (Optional) Activation layer information in case of a fused activation.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo         *input,
                                 const BIITensorInfo         *weights,
                                 const BIITensorInfo         *bias,
                                 const BIITensorInfo         *output,
                                 const BIPadStrideInfo       &conv_info,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        // Inherited methods overridden:
        void run();

    private:
        struct Impl;
        std::shared_ptr<BIIMemoryManager> _memory_manager;
        std::unique_ptr<Impl>             _impl;
    };

} // namespace BatmanInfer
