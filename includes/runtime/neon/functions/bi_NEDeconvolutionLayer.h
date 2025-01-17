//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/cpp/functions/bi_cpp_upsample.h"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_memory_manager.hpp"
#include "runtime/bi_memory_group.hpp"
#include "runtime/bi_memory_manager_on_demand.hpp"
#include "runtime/neon/functions/bi_NEConvolutionLayer.h"
#include "runtime/neon/functions/bi_NEDirectConvolutionLayer.h"
#include "runtime/neon/functions/bi_ne_reverse.h"
#include "runtime/bi_tensor.hpp"

#include <memory>

namespace BatmanInfer {

    /** Function to run the deconvolution layer.
     *
     * Deconvolution Layer is the backward pass of Convolution Layer. First we transform the input depending on the stride and pad info and then perfrom a 1x1
     * convolution pass. Input stride defines how many zeroes we should put between each element of the input, pad is the amount of padding and finaly a is a user
     * specified value where a < stride - 1 that increases the padding top and right of the input image.
     *
     *  The relation between input to output is as follows:
     *  \f[
     *       width\_output = (width\_input - 1) \cdot stride\_x - 2 \cdot padding\_x + kernel\_x
     *  \f]
     *  \f[
     *       height\_output = (height\_input - 1) \cdot stride\_y - 2 \cdot padding\_y + kernel\_y
     *  \f]
     *
     *  where
     *      width is the size of the first input dimension.
     *      height is the size of the second input dimension.
     *      width_output is the size of the first output dimension.
     *      height_output is the size of the second output dimension.
     *      kernel_x and kernel_y are the convolution sizes in x and y.
     *      stride_x and stride_y is the input stride of the first and second dimension.
     *
     * The weights used by Deconvolution are supposed to be the same as the ones used for Convolution. Therefore, it will be necessary to use the weights in the
     * reverse order to perform an actual convolution. This is achieved by using @ref NEReverse.
     *
     * This function calls the following kernels/functions:
     *
     * -# @ref CPPUpsample
     * -# @ref NEConvolutionLayer
     * -# @ref NEReverse
     *
     */
    class BINEDeconvolutionLayer : public BIIFunction
    {
    public:
        /** Constructor */
        BINEDeconvolutionLayer(std::shared_ptr<BIIMemoryManager> memory_manager);
        BINEDeconvolutionLayer() : BINEDeconvolutionLayer(BIMemoryManagerOnDemand::make_default())
        {
        }
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEDeconvolutionLayer(const BINEDeconvolutionLayer &) = delete;
        /** Default move constructor */
        BINEDeconvolutionLayer(BINEDeconvolutionLayer &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEDeconvolutionLayer &operator=(const BINEDeconvolutionLayer &) = delete;
        /** Default move assignment operator */
        BINEDeconvolutionLayer &operator=(BINEDeconvolutionLayer &&) = default;
        /** Default destructor */
        ~BINEDeconvolutionLayer() = default;

        /** Set the input, weights, biases and output tensors.
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src0           |src1               |src2   |dst            |
         * |:--------------|:------------------|:------|:--------------|
         * |F16            |F16                |F16    |F16            |
         * |F32            |F32                |F32    |F32            |
         * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
         * |QASYMM8        |QSYMM8_PER_CHANNEL |S32    |QASYMM8        |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
         * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32    |QASYMM8_SIGNED |
         *
         * @param[in,out] input            Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs.
         *                                 Data types supported: F32/F16/QASYMM8/QASYMM8_SIGNED.
         * @param[in]     weights          The 4d weights with dimensions [width, height, IFM, OFM].
         *                                 Data type supported: Same as @p input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
         * @param[in]     bias             Optional, ignored if NULL. The biases have one dimension.
         *                                 Data type supported: Data types supported: S32 for QASYMM8/QASYMM8_SIGNED input, F32 for F32 input, F16 for F16 input.
         * @param[out]    output           Output tensor. The output has the same number of dimensions as the @p input.
         * @param[in]     info             Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo.
         * @param[in]     enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
         *                                 available which may introduce a drop of accuracy as well. Default is false
         * @param[in]     weights_info     (Optional) Specifies the weight format. Default is unspecified. This parameter can be used to specify the weight format that is optimal for
         *                                 the GEMM convolution.
         *
         */
        void configure(BIITensor             *input,
                       const BIITensor       *weights,
                       const BIITensor       *bias,
                       BIITensor             *output,
                       const BIPadStrideInfo &info,
                       bool                   enable_fast_math = false,
                       const BIWeightsInfo   &weights_info     = BIWeightsInfo());
        /** Static function to check if given info will lead to a valid configuration of @ref NEDeconvolutionLayer
         *
         * @param[in] input            Input tensor info. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs.
         *                             Data types supported: F32/F16/QASYMM8/QASYMM8_SIGNED.
         * @param[in] weights          The 4d weights info with dimensions [width, height, IFM, OFM].
         *                             Data type supported: Same as @p input, also could be QSYMM8_PER_CHANNEL if input is QASYMM8/QASYMM8_SIGNED.
         * @param[in] bias             (Optional) The biases have one dimension. Data type supported: Data types supported: S32 for QASYMM8/QASYMM8_SIGNED input, F32 for F32 input, F16 for F16 input.
         * @param[in] output           Output tensor info. The output has the same number of dimensions as the @p input.
         * @param[in] info             Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo.
         * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
         *                             available which may introduce a drop of accuracy as well. Default is false
         * @param[in] weights_info     (Optional) Specifies the weight format. Default is unspecified. This parameter can be used to specify the weight format that is optimal for
         *                             the GEMM convolution.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo   *input,
                                 const BIITensorInfo   *weights,
                                 const BIITensorInfo   *bias,
                                 const BIITensorInfo   *output,
                                 const BIPadStrideInfo &info,
                                 bool                   enable_fast_math = false,
                                 const BIWeightsInfo   &weights_info     = BIWeightsInfo());

        // Inherited methods overridden:
        void run() override;
        void prepare() override;

    private:
        BIMemoryGroup        _memory_group;
        BINEConvolutionLayer _conv_f;
        BICPPUpsample        _upsample_f;
        BINEReverse          _flip_weights;
        BITensor             _scaled_output;
        BITensor             _weights_flipped;
        BITensor             _flip_axis;
        const BIITensor     *_original_weights;
        BIITensor           *_input;
        BIPadStrideInfo      _info;
        bool                 _is_prepared;
        bool                 _do_upsampling;
    };

} // namespace BatmanInfer
