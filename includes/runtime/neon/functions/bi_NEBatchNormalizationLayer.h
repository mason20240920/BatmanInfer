//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/bi_i_function.hpp"
#include "runtime/neon/functions/bi_NEActivationLayer.h"

namespace BatmanInfer {

    class BIITensor;
    class BINEBatchNormalizationLayerKernel;

    /** Basic function to run NENormalizationLayerKernel and simulate a batch normalization layer.
     *
     * Batch normalization is calculated by:
     * @f[ out_i = \gamma * (\frac{in_i - \mu_{B}}{\sqrt{\sigma^2_{B} + \epsilon}}) + \beta \equiv BN_{\gamma,\beta}(in_i) @f]
     *
     */
    class BINEBatchNormalizationLayer : public BIIFunction
    {
    public:
        /** Constructor */
        BINEBatchNormalizationLayer();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEBatchNormalizationLayer(const BINEBatchNormalizationLayer &) = delete;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEBatchNormalizationLayer &operator=(const BINEBatchNormalizationLayer &) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEBatchNormalizationLayer(BINEBatchNormalizationLayer &&) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEBatchNormalizationLayer &operator=(BINEBatchNormalizationLayer &&) = delete;
        /** Default destructor */
        ~BINEBatchNormalizationLayer();
        /** Set the input and output tensors.
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src            |dst            |
         * |:--------------|:--------------|
         * |F32            |F32            |
         * |F16            |F16            |
         *
         * @note If the output tensor is a nullptr or is equal to the input, the batch normalization function will be performed in-place
         *
         * @param[in, out] input    Source tensor. In case of @p output tensor = nullptr, this tensor will store the result.
         *                          3 lower dimensions represent a single input with dimensions [width, height, FM].
         *                          The rest are optional and used for representing batches. Data types supported: F16/F32.
         * @param[out]     output   Destination tensor. Output will have the same number of dimensions as input. Data type supported: same as @p input
         * @param[in]      mean     Mean values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
         * @param[in]      var      Variance values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
         * @param[in]      beta     (Optional) Beta values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for beta is 0. Data types supported: Same as @p input
         * @param[in]      gamma    (Optional) Gamma values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for gamma is 1. Data types supported: Same as @p input
         * @param[in]      epsilon  (Optional) Small value to avoid division with zero. Default value is 0.001f.
         * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
         */
        void configure(BIITensor            *input,
                       BIITensor            *output,
                       const BIITensor      *mean,
                       const BIITensor      *var,
                       const BIITensor      *beta     = nullptr,
                       const BIITensor      *gamma    = nullptr,
                       float                 epsilon  = 0.001f,
                       BIActivationLayerInfo act_info = BIActivationLayerInfo());
        /** Static function to check if given info will lead to a valid configuration of @ref NEBatchNormalizationLayer
         *
         * @param[in] input    Source tensor info. In case of @p output tensor = nullptr, this tensor will store the result.
         *                     3 lower dimensions represent a single input with dimensions [width, height, FM].
         *                     The rest are optional and used for representing batches. Data types supported: F16/F32.
         * @param[in] output   Destination tensor info. Output will have the same number of dimensions as input. Data type supported: same as @p input
         * @param[in] mean     Mean values tensor info. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
         * @param[in] var      Variance values tensor info. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
         * @param[in] beta     (Optional) Beta values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for beta is 0. Data types supported: Same as @p input
         * @param[in] gamma    (Optional) Gamma values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for gamma is 1. Data types supported: Same as @p input
         * @param[in] epsilon  (Optional) Small value to avoid division with zero. Default value is 0.001f.
         * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo  *input,
                                 const BIITensorInfo  *output,
                                 const BIITensorInfo  *mean,
                                 const BIITensorInfo  *var,
                                 const BIITensorInfo  *beta     = nullptr,
                                 const BIITensorInfo  *gamma    = nullptr,
                                 float                 epsilon  = 0.001f,
                                 BIActivationLayerInfo act_info = BIActivationLayerInfo());

        // Inherited methods overridden:
        void run() override;

    private:
        std::unique_ptr<BINEBatchNormalizationLayerKernel> _norm_kernel; /**< Batch normalization layer kernel */
    };

} // namespace BatmanInfer
