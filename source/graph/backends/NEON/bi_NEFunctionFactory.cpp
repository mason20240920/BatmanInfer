//
// Created by holynova on 2025/1/16.
//

#include "graph/backends/NEON/bi_NEFunctionFactory.h"

#include "graph/backends/bi_FunctionHelpers.h"
#include "graph/backends/bi_utils.h"
#include "graph/bi_graph.h"
#include "graph/bi_graphContext.h"
#include "graph/bi_logger.h"
#include "graph/nodes/Nodes.h"
#include "graph/bi_typePrinter.h"
#include "runtime/cpp/bi_cpp_functions.h"
#include "runtime/neon/bi_ne_functions.h"
#include "runtime/bi_tensor.hpp"

using namespace BatmanInfer::utils::cast;

namespace BatmanInfer {

namespace graph {

namespace backends {

    /** Target specific information structure used to pass information to the layer templates */
    struct BINETargetInfo
    {
        using TensorType         = BatmanInfer::BIITensor;
        using SrcTensorType      = const BatmanInfer::BIITensor;
        using TensorConcreteType = BatmanInfer::BITensor;
        static BITarget TargetType;
    };

    BITarget BINETargetInfo::TargetType = BITarget::NEON;

    /** Collection of CPU convolution functions */
    struct BINEConvolutionLayerFunctions
    {
        using GenericConvolutionLayer  = BINEConvolutionLayer;
        using GEMMConvolutionLayer     = BINEGEMMConvolutionLayer;
        using DirectConvolutionLayer   = BINEDirectConvolutionLayer;
        using WinogradConvolutionLayer = BINEWinogradConvolutionLayer;
    };

    /** Collection of CPU element-wise functions */
    struct BINEEltwiseFunctions
    {
        using Addition       = BINEArithmeticAddition;
        using Subtraction    = BINEArithmeticSubtraction;
        using Multiplication = BINEPixelWiseMultiplication;
        using Maximum        = BINEElementwiseMax;
        using Division       = BINEElementwiseDivision;
    };

    /** Collection of CPU unary element-wise functions */
    // struct BINEUnaryEltwiseFunctions
    // {
    //     using Exp = NEExpLayer;
    // };

    /** Function and tensor types to be used inside a fused convolution/batch normalization layer */
    // struct BINEFusedLayerTypes
    // {
    //     using ConvolutionLayer          = NEConvolutionLayer;
    //     using DepthwiseConvolutionLayer = NEDepthwiseConvolutionLayer;
    //     using FuseBatchNormalization    = NEFuseBatchNormalization;
    // };

    namespace detail {

        // template <>
        // std::unique_ptr<BIIFunction> create_normalization_layer<NENormalizationLayer, NETargetInfo>(
        //     NormalizationLayerNode &node,
        //     BIGraphContext         &ctx)
        // {
        //     validate_node<NETargetInfo>(node, 1 /* expected inputs */, 1 /* expected outputs */);

            // Extract IO and info
        //     BINETargetInfo::TensorType    *input     = get_backing_tensor<BINETargetInfo>(node.input(0));
        //     BINETargetInfo::TensorType    *output    = get_backing_tensor<BINETargetInfo>(node.output(0));
        //     const BINormalizationLayerInfo norm_info = node.normalization_info();
        //     BI_COMPUTE_ERROR_ON(input == nullptr);
        //     BI_COMPUTE_ERROR_ON(output == nullptr);

            // Create and configure function
        //     auto func = std::make_unique<NENormalizationLayer>(get_memory_manager(ctx, BINETargetInfo::TargetType));
        //     func->configure(input, output, norm_info);

            // Log info
        //     BI_COMPUTE_LOG_GRAPH_INFO("Instantiated "
        //                               << node.name() << " Type: " << node.type() << " Target: " << NETargetInfo::TargetType
        //                               << " Data Type: " << input->info()->data_type() << " Input shape: "
        //                               << input->info()->tensor_shape() << " Output shape: " << output->info()->tensor_shape()
        //                               << " Normalization info: " << norm_info.type() << std::endl);

        //     return func;
        // }

    } // namespace detail

    std::unique_ptr<BIIFunction> BINEFunctionFactory::create(BIINode *node, BIGraphContext &ctx)
    {
        if (node == nullptr)
        {
            return nullptr;
        }

        BINodeType type = node->type();
        switch (type)
        {
            case BINodeType::ActivationLayer:
                return detail::create_activation_layer<BINEActivationLayer, BINETargetInfo>(
                    *polymorphic_downcast<ActivationLayerNode *>(node));
            case BINodeType::ArgMinMaxLayer:
                return detail::create_arg_min_max_layer<BINEArgMinMaxLayer, BINETargetInfo>(
                    *polymorphic_downcast<ArgMinMaxLayerNode *>(node));
            case BINodeType::BatchNormalizationLayer:
                return detail::create_batch_normalization_layer<BINEBatchNormalizationLayer, BINETargetInfo>(
                    *polymorphic_downcast<BatchNormalizationLayerNode *>(node));
            case BINodeType::ChannelShuffleLayer:
                return detail::create_channel_shuffle_layer<BINEChannelShuffleLayer, BINETargetInfo>(
                    *polymorphic_downcast<ChannelShuffleLayerNode *>(node));
            case BINodeType::ConvolutionLayer:
                return detail::create_convolution_layer<BINEConvolutionLayerFunctions, BINETargetInfo>(
                    *polymorphic_downcast<ConvolutionLayerNode *>(node), ctx);
            case BINodeType::DepthToSpaceLayer:
                return detail::create_depth_to_space_layer<BINEDepthToSpaceLayer, BINETargetInfo>(
                    *polymorphic_downcast<DepthToSpaceLayerNode *>(node));
            case BINodeType::DeconvolutionLayer:
                return detail::create_deconvolution_layer<BINEDeconvolutionLayer, BINETargetInfo>(
                    *polymorphic_downcast<DeconvolutionLayerNode *>(node), ctx);
            case BINodeType::ConcatenateLayer:
                return detail::create_concatenate_layer<BINEConcatenateLayer, BINETargetInfo>(
                    *polymorphic_downcast<ConcatenateLayerNode *>(node));
            case BINodeType::DepthwiseConvolutionLayer:
                return detail::create_depthwise_convolution_layer<BINEDepthwiseConvolutionLayer, BINETargetInfo>(
                    *polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node));
            case BINodeType::DequantizationLayer:
                return detail::create_dequantization_layer<BINEDequantizationLayer, BINETargetInfo>(
                    *polymorphic_downcast<DequantizationLayerNode *>(node));
            // case BINodeType::DetectionOutputLayer:
            //     return detail::create_detection_output_layer<CPPDetectionOutputLayer, BINETargetInfo>(
            //         *polymorphic_downcast<DetectionOutputLayerNode *>(node));
            // case BINodeType::DetectionPostProcessLayer:
            //     return detail::create_detection_post_process_layer<NEDetectionPostProcessLayer, BIBINETargetInfo>(
            //         *polymorphic_downcast<DetectionPostProcessLayerNode *>(node));
            case BINodeType::EltwiseLayer:
                return detail::create_eltwise_layer<BINEEltwiseFunctions, BINETargetInfo>(
                    *polymorphic_downcast<EltwiseLayerNode *>(node));
            // case BINodeType::UnaryEltwiseLayer:
            //     return detail::create_unary_eltwise_layer<NEUnaryEltwiseFunctions, BINETargetInfo>(
            //         *polymorphic_downcast<UnaryEltwiseLayerNode *>(node));
            case BINodeType::FlattenLayer:
                return detail::create_flatten_layer<BINEFlattenLayer, BINETargetInfo>(
                    *polymorphic_downcast<FlattenLayerNode *>(node));
            case BINodeType::FullyConnectedLayer:
                return detail::create_fully_connected_layer<BINEFullyConnectedLayer, BINETargetInfo>(
                    *polymorphic_downcast<FullyConnectedLayerNode *>(node), ctx);
            // case BINodeType::FusedConvolutionBatchNormalizationLayer:
            //     return detail::create_fused_convolution_batch_normalization_layer<NEFusedLayerTypes, BINETargetInfo>(
            //         *polymorphic_downcast<FusedConvolutionBatchNormalizationNode *>(node), ctx);
            // case BINodeType::FusedDepthwiseConvolutionBatchNormalizationLayer:
            //     return detail::create_fused_depthwise_convolution_batch_normalization_layer<NEFusedLayerTypes,
            //                                                                                 BINETargetInfo>(
            //         *polymorphic_downcast<FusedDepthwiseConvolutionBatchNormalizationNode *>(node), ctx);
            // case BINodeType::L2NormalizeLayer:
            //     return detail::create_l2_normalize_layer<NEL2NormalizeLayer, BINETargetInfo>(
            //         *polymorphic_downcast<L2NormalizeLayerNode *>(node), ctx);
            // case BINodeType::NormalizationLayer:
            //     return detail::create_normalization_layer<NENormalizationLayer, BINETargetInfo>(
            //         *polymorphic_downcast<NormalizationLayerNode *>(node), ctx);
            // case BINodeType::PadLayer:
            //     return detail::create_pad_layer<NEPadLayer, BINETargetInfo>(*polymorphic_downcast<PadLayerNode *>(node));
            case BINodeType::PermuteLayer:
                return detail::create_permute_layer<BINEPermute, BINETargetInfo>(
                    *polymorphic_downcast<PermuteLayerNode *>(node));
            // case BINodeType::PoolingLayer:
            //     return detail::create_pooling_layer<NEPoolingLayer, BINETargetInfo>(
            //         *polymorphic_downcast<PoolingLayerNode *>(node));
            case BINodeType::PReluLayer:
                return detail::create_prelu_layer<BINEPReluLayer, BINETargetInfo>(
                    *polymorphic_downcast<PReluLayerNode *>(node));
            case BINodeType::PrintLayer:
                return detail::create_print_layer<BINETargetInfo>(*polymorphic_downcast<PrintLayerNode *>(node));
            // case BINodeType::PriorBoxLayer:
            //     return detail::create_priorbox_layer<NEPriorBoxLayer, BINETargetInfo>(
            //         *polymorphic_downcast<PriorBoxLayerNode *>(node));
            case BINodeType::QuantizationLayer:
                return detail::create_quantization_layer<BINEQuantizationLayer, BINETargetInfo>(
                    *polymorphic_downcast<QuantizationLayerNode *>(node));
            // case BINodeType::ReductionOperationLayer:
            //     return detail::create_reduction_operation_layer<NEReductionOperation, BINETargetInfo>(
            //         *polymorphic_downcast<ReductionLayerNode *>(node), ctx);
            // case BINodeType::ReorgLayer:
            //     return detail::create_reorg_layer<NEReorgLayer, BINETargetInfo>(
            //         *polymorphic_downcast<ReorgLayerNode *>(node));
            case BINodeType::ReshapeLayer:
                return detail::create_reshape_layer<BINEReshapeLayer, BINETargetInfo>(
                    *polymorphic_downcast<ReshapeLayerNode *>(node));
            case BINodeType::ResizeLayer:
                return detail::create_resize_layer<BINEScale, BINETargetInfo>(*polymorphic_downcast<ResizeLayerNode *>(node));
            case BINodeType::SliceLayer:
                return detail::create_slice_layer<BINESlice, BINETargetInfo>(*polymorphic_downcast<SliceLayerNode *>(node));
            case BINodeType::SoftmaxLayer:
                return detail::create_softmax_layer<BINESoftmaxLayer, BINETargetInfo>(
                    *polymorphic_downcast<SoftmaxLayerNode *>(node), ctx);
            // case BINodeType::StackLayer:
            //     return detail::create_stack_layer<NEStackLayer, BINETargetInfo>(
            //         *polymorphic_downcast<StackLayerNode *>(node));
            // case BINodeType::StridedSliceLayer:
            //     return detail::create_strided_slice_layer<NEStridedSlice, BINETargetInfo>(
            //         *polymorphic_downcast<StridedSliceLayerNode *>(node));
            default:
                return nullptr;
        }
    }

} // namespace backends

} // namespace graph

} // namespace BatmanInfer
