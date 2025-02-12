//
// Created by holynova on 2025/2/4.
//

#pragma once

#include "graph/bi_types.h"

namespace BatmanInfer {

namespace graph {

namespace frontend {

    // Import types for graph
    using graph::BIDataLayout;
    using graph::BIDataLayoutDimension;
    using graph::BIDataType;
    using BatmanInfer::PermutationVector;
    using graph::BITensorShape;

    using graph::BIActivationLayerInfo;
    using graph::BIConvolutionMethod;
    using graph::BIDepthwiseConvolutionMethod;
    using graph::BIDimensionRoundingType;
    using graph::BIEltwiseOperation;
    using graph::BIFastMathHint;
    using graph::BIFullyConnectedLayerInfo;
    using graph::BIGraphConfig;
    using BatmanInfer::BIInterpolationPolicy;
    using BatmanInfer::BINormalizationLayerInfo;
    using BatmanInfer::BINormType;
    using graph::BIPadStrideInfo;
    using BatmanInfer::BIPoolingLayerInfo;
    using BatmanInfer::BIPoolingType;
    using graph::Size2D;
    using graph::BITarget;
    using graph::BITensorDescriptor;

    typedef struct _FrontendHints_
    {
        /**< Target execution hint */
        BITarget target_hint = {BITarget::UNSPECIFIED};
        /**< Convolution method hint */
        BIConvolutionMethod convolution_method_hint = {BIConvolutionMethod::Default};
        /**< Depthwise Convolution method hint */
        BIDepthwiseConvolutionMethod depthwise_convolution_method_hint = { BIDepthwiseConvolutionMethod::Default};
        /**< Fast math hint */
        BIFastMathHint fast_math_hint = {BIFastMathHint::Disabled};
    }FrontendHints;

} // namespace frontend

} // namespace graph

} // namespace BatmanInfer
