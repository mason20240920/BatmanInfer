//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "function_info/bi_activationLayerInfo.h"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_memory_manager.hpp"
#include "runtime/bi_i_weights_manager.hpp"
#include "runtime/bi_memory_group.hpp"
#include "runtime/bi_memory_manager_on_demand.hpp"

#include <memory>

namespace BatmanInfer {

    class BIITensor;
    class BIITensorInfo;

    /** Basic function to compute the convolution layer. This function calls the following kernels/functions:
     *
     * -# cpu::CpuGemmConv2d
     *
     */
    class BINEGEMMConvolutionLayer : public BIIFunction
    {
    public:
        /** Constructor */
        BINEGEMMConvolutionLayer(const std::shared_ptr<BIIMemoryManager> &memory_manager,
                                 BIIWeightsManager                       *weights_manager = nullptr);
        BINEGEMMConvolutionLayer() : BINEGEMMConvolutionLayer(BIMemoryManagerOnDemand::make_default())
        {
        }
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEGEMMConvolutionLayer(const BINEGEMMConvolutionLayer &) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEGEMMConvolutionLayer(BINEGEMMConvolutionLayer &&) = delete;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEGEMMConvolutionLayer &operator=(const BINEGEMMConvolutionLayer &) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEGEMMConvolutionLayer &operator=(BINEGEMMConvolutionLayer &&) = delete;
        /** Default destructor */
        ~BINEGEMMConvolutionLayer();
        /** Set the input and output tensors.
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src0           |src1               |src2     |dst            |
         * |:--------------|:------------------|:--------|:--------------|
         * |F16            |F16                |F16      |F16            |
         * |F32            |F32                |F32      |F32            |
         * |BFLOAT16       |BFLOAT16           |BFLOAT16 |BFLOAT16       |
         * |QASYMM8        |QASYMM8            |S32      |QASYMM8        |
         * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
         * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
         *
         * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
         *                              while every optional dimension from 4 and above represent a batch of inputs.
         *                              Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
         * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
         *                              Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
         * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
         *                              Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
         * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
         *                              Data types supported: Same as @p input.
         * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
         * @param[in]  weights_info     Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
         *                              tensor has also been transposed with cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p input.
         * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
         * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
         *                              available which may introduce a drop of accuracy as well. Default is false
         * @param[in]  num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
         */
        void configure(const BIITensor             *input,
                       const BIITensor             *weights,
                       const BIITensor             *biases,
                       BIITensor                   *output,
                       const BIPadStrideInfo       &conv_info,
                       const BIWeightsInfo         &weights_info     = BIWeightsInfo(),
                       const Size2D                &dilation         = Size2D(1U, 1U),
                       const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                       bool                         enable_fast_math = false,
                       unsigned int                 num_groups       = 1);
        /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMConvolutionLayer
         *
         * @param[in] input            Source tensor info. 3 lower dimensions represent a single input [width, height, IFM],
         *                             while every optional dimension from 4 and above represent a batch of inputs.
         *                             Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
         * @param[in] weights          Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
         *                             Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
         * @param[in] biases           Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
         *                             Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
         * @param[in] output           Destination tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
         *                             Data types supported: Same as @p input.
         * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
         * @param[in] weights_info     Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
         *                             tensor has also been transposed with cpu::kernels::CpuGemmTranspose1xWKernel. Data type supported: Same as @p input.
         * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         * @param[in] act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
         * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
         *                             available which may introduce a drop of accuracy as well. Default is false
         * @param[in] num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo         *input,
                                 const BIITensorInfo         *weights,
                                 const BIITensorInfo         *biases,
                                 const BIITensorInfo         *output,
                                 const BIPadStrideInfo       &conv_info,
                                 const BIWeightsInfo         &weights_info     = BIWeightsInfo(),
                                 const Size2D                &dilation         = Size2D(1U, 1U),
                                 const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                                 bool                         enable_fast_math = false,
                                 unsigned int                 num_groups       = 1);

        /** Static function to check if there is an optimized version of
         * GEMM available for the input parameters.
         *
         * The method is intended to be used to find out the optimal
         * memory layout to be used for the weights tensor when running
         * variable weights execution.
         *
         * The user can query the database of optimised kernels in
         * arm_gemm by specifying one of the enumerations of
         * arm_compute::WeightFormat in the weight_format field of the input
         * parameter weights_info. In case of success, the method
         * writes the expected format in the output parameter
         * expected_weight_format. The expected_weight_format can than be
         * used in the configure method of the class for retrieving the
         * best optimal kernel.
         *
         * Use case one - query for a specific format:
         *
         *     WeightInfo weights_info(..., arm_compute::WeightFormat::OHWIo4, ...); // Set the value of the input query.
         *     if (NEGEMMConvolutionlayer::has_opt_impl(WeightFormat(), ...., weights_info, ...))
         *     {
         *       auto conv = std::unique_ptr<NEGEMMConvolutionlayer>();
         *       conv->configure(..., weights_info, ...);  // uses the same WeightFormat the user wanted originally, OHWYo4.
         *       conv->run(...);
         *     }
         *
         * Use case two - query for any format that would be optimal for the GEMM to execute:
         *
         *     WeightInfo weights_info(..., arm_compute::WeightFormat::ANY, ...); // Set the value of the input query.
         *     arm_compute::WeightFormat expected_wf;
         *     if (NEGEMMConvolutionlayer::has_opt_impl(expected_wf, ...., weights_info, ...))
         *     {
         *       auto conv = std::unique_ptr<NEGEMMConvolutionlayer>();
         *       // ... code to convert the layout of the weights tensor to the layout returned by has_opt_impl
         *       WeightInfo new_weights_info(..., expected_wf, ...); // Set the value of the WeightFormat returned by has_opt_impl.
         *       conv->configure(..., new_weights_info, ...);
         *       conv->run(...);
         *     }
         *
         * Notice that a GEMM configured with a WeightFormat other than
         * UNSPECIFIED will run GEMM with variable weights mode.
         *
         * @param[out] expected_weight_format The arm_compute::WeightFormat expected by the kernel.
         * @param[in]  src                    Source tensor info.
         * @param[in]  weights                Weights tensor info.
         * @param[in]  biases                 Biases tensor info. Shared biases supported.
         * @param[in]  dst                    Destination tensor info.
         * @param[in]  conv_info              Contains padding and stride information described in @ref PadStrideInfo.
         * @param[in]  weights_info           (optional) Specifies additional configuration parameters for the weights of the GEMM computation.
         * @param[in]  dilation               (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         * @param[in]  act_info               (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported. And no activation (i.e. Linear) which is the default value.
         * @param[in]  enable_fast_math       (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
         *
         * @return a Status
         */
        static BIStatus has_opt_impl(BatmanInfer::BIWeightFormat &expected_weight_format,
                                     const BIITensorInfo         *src,
                                     const BIITensorInfo         *weights,
                                     const BIITensorInfo         *biases,
                                     const BIITensorInfo         *dst,
                                     const BIPadStrideInfo       &conv_info,
                                     const BIWeightsInfo         &weights_info     = BIWeightsInfo(),
                                     const Size2D                &dilation         = Size2D(1U, 1U),
                                     const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                                     bool                         enable_fast_math = false);

        /** Update of quantization information at the run stage for convolution so that the quantization multipliers can be properly calculated.
         *
         * Quantization information is usually required at configure time, that is, when we call configure(). However, this information is not always available at this stage.
         * Quantization parameters are sometimes only available in the run() stage. But at this stage it is too late, as the operator was already configured everything based on the
         * wrong quantization parameters. One could call configure() again when the correct information is available, but that is a huge overhead. Therefore, this solution
         * that correctly and efficiently propagates the quantization information (multipliers and offsets) to the kernel.
         *
         * Example:
         *
         * op->configure(...)
         * make sure to set the correct parameters in the tensors by updating QuantizationInfo before calling update_quantization_parameters.
         * op->update_quantization_parameters()
         * op->run(...)
         *
         * Please have a look at examples/neon_gemm_qasymm8_signed.cpp for an example realization.
         *
         * This API is experimental and it can be changed in the future.
         */
        void update_quantization_parameters();

        // Inherited methods overridden:
        void run() override;
        void prepare() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
