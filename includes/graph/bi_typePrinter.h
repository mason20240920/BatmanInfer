//
// Created by holynova on 2025/1/10.
//

#ifndef BATMANINFER_GRAPH_BI_TYPEPRINTER_H
#define BATMANINFER_GRAPH_BI_TYPEPRINTER_H

#include "data/core/bi_error.h"
#include "data/core/bi_types.hpp"
#include "graph/bi_types.h"

#include "utils/bi_type_printer.hpp"

namespace BatmanInfer {

namespace graph {

    /** Formatted output of the Target. */
    inline ::std::ostream &operator<<(::std::ostream &os, const BITarget &target)
    {
        switch (target)
        {
        case BITarget::UNSPECIFIED:
            os << "UNSPECIFIED";
            break;
        case BITarget::NEON:
            os << "Neon";
            break;
        case BITarget::CL:
            os << "CL";
            break;
        case BITarget::CLVK:
            os << "CLVK";
            break;
        default:
            BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }

    inline ::std::ostream &operator<<(::std::ostream &os, const BINodeType &node_type)
    {
        switch (node_type)
        {
            case BINodeType::ActivationLayer:
                os << "ActivationLayer";
                break;
            case BINodeType::ArgMinMaxLayer:
                os << "ArgMinMaxLayer";
                break;
            case BINodeType::BatchNormalizationLayer:
                os << "BatchNormalizationLayer";
                break;
            case BINodeType::BoundingBoxTransformLayer:
                os << "BoundingBoxTransformLayer";
                break;
            case BINodeType::ChannelShuffleLayer:
                os << "ChannelShuffleLayer";
                break;
            case BINodeType::ConcatenateLayer:
                os << "ConcatenateLayer";
                break;
            case BINodeType::ConvolutionLayer:
                os << "ConvolutionLayer";
                break;
            case BINodeType::DeconvolutionLayer:
                os << "DeconvolutionLayer";
                break;
            case BINodeType::DepthToSpaceLayer:
                os << "DepthToSpaceLayer";
                break;
            case BINodeType::DequantizationLayer:
                os << "DequantizationLayer";
                break;
            case BINodeType::DetectionOutputLayer:
                os << "DetectionOutputLayer";
                break;
            case BINodeType::DetectionPostProcessLayer:
                os << "DetectionPostProcessLayer";
                break;
            case BINodeType::DepthwiseConvolutionLayer:
                os << "DepthwiseConvolutionLayer";
                break;
            case BINodeType::EltwiseLayer:
                os << "EltwiseLayer";
                break;
            case BINodeType::UnaryEltwiseLayer:
                os << "UnaryEltwiseLayer";
                break;
            case BINodeType::FlattenLayer:
                os << "FlattenLayer";
                break;
            case BINodeType::FullyConnectedLayer:
                os << "FullyConnectedLayer";
                break;
            case BINodeType::FusedConvolutionBatchNormalizationLayer:
                os << "FusedConvolutionBatchNormalizationLayer";
                break;
            case BINodeType::FusedDepthwiseConvolutionBatchNormalizationLayer:
                os << "FusedDepthwiseConvolutionBatchNormalizationLayer";
                break;
            case BINodeType::GenerateProposalsLayer:
                os << "GenerateProposalsLayer";
                break;
            case BINodeType::L2NormalizeLayer:
                os << "L2NormalizeLayer";
                break;
            case BINodeType::NormalizationLayer:
                os << "NormalizationLayer";
                break;
            case BINodeType::NormalizePlanarYUVLayer:
                os << "NormalizePlanarYUVLayer";
                break;
            case BINodeType::PadLayer:
                os << "PadLayer";
                break;
            case BINodeType::PermuteLayer:
                os << "PermuteLayer";
                break;
            case BINodeType::PoolingLayer:
                os << "PoolingLayer";
                break;
            case BINodeType::PReluLayer:
                os << "PReluLayer";
                break;
            case BINodeType::PrintLayer:
                os << "PrintLayer";
                break;
            case BINodeType::PriorBoxLayer:
                os << "PriorBoxLayer";
                break;
            case BINodeType::QuantizationLayer:
                os << "QuantizationLayer";
                break;
            case BINodeType::ReductionOperationLayer:
                os << "ReductionOperationLayer";
                break;
            case BINodeType::ReorgLayer:
                os << "ReorgLayer";
                break;
            case BINodeType::ReshapeLayer:
                os << "ReshapeLayer";
                break;
            case BINodeType::ResizeLayer:
                os << "ResizeLayer";
                break;
            case BINodeType::ROIAlignLayer:
                os << "ROIAlignLayer";
                break;
            case BINodeType::SoftmaxLayer:
                os << "SoftmaxLayer";
                break;
            case BINodeType::SliceLayer:
                os << "SliceLayer";
                break;
            case BINodeType::SplitLayer:
                os << "SplitLayer";
                break;
            case BINodeType::StackLayer:
                os << "StackLayer";
                break;
            case BINodeType::StridedSliceLayer:
                os << "StridedSliceLayer";
                break;
            case BINodeType::UpsampleLayer:
                os << "UpsampleLayer";
                break;
            case BINodeType::Input:
                os << "Input";
                break;
            case BINodeType::Output:
                os << "Output";
                break;
            case BINodeType::Const:
                os << "Const";
                break;
            case BINodeType::Dummy:
                os << "Dummy";
                break;
            default:
                BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }

    /** Formatted output of the EltwiseOperation type. */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIEltwiseOperation &eltwise_op)
    {
        switch (eltwise_op)
        {
        case BIEltwiseOperation::Add:
            os << "Add";
            break;
        case BIEltwiseOperation::Mul:
            os << "Mul";
            break;
        case BIEltwiseOperation::Sub:
            os << "Sub";
            break;
        case BIEltwiseOperation::Div:
            os << "Div";
            break;
        default:
            BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }

    /** Formatted output of the ConvolutionMethod type. */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIConvolutionMethod &method)
    {
        switch (method)
        {
        case BIConvolutionMethod::Default:
            os << "Default";
            break;
        case BIConvolutionMethod::Direct:
            os << "Direct";
            break;
        case BIConvolutionMethod::GEMM:
            os << "GEMM";
            break;
        case BIConvolutionMethod::Winograd:
            os << "Winograd";
            break;
        default:
            BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }

    /** Formatted output of the FastMathHint type. */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIFastMathHint &hint)
    {
        switch (hint)
        {
        case BIFastMathHint::Enabled:
            os << "Enabled";
            break;
        case BIFastMathHint::Disabled:
            os << "Disabled";
            break;
        default:
            BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }

    /** Formatted output of the DepthwiseConvolutionMethod type. */
    inline ::std::ostream &operator<<(::std::ostream &os, const BIDepthwiseConvolutionMethod &method)
    {
        switch (method)
        {
        case BIDepthwiseConvolutionMethod::Default:
            os << "DEFAULT";
            break;
        case BIDepthwiseConvolutionMethod::Optimized3x3:
            os << "Optimized3x3";
            break;
        default:
            BI_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        return os;
    }

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_TYPEPRINTER_H
