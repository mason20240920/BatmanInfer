//
// Created by holynova on 2025/1/15.
//

#pragma once

#include "data/core/bi_error.h"
#include "data/core/bi_helpers.hpp"
#include "data/core/bi_i_tensor_info.hpp"
#include "graph/bi_logger.h"
#include "graph/nodes/Nodes.h"
#include "graph/bi_tensor.h"
#include "graph/bi_types.h"

namespace BatmanInfer {

namespace graph {

namespace backends {

namespace detail {

/** Returns backing tensor info of a given tensor
 *
 * @param[in] tensor Tensor to extract the backing tensor from
 *
 * @return Backing tensor tensor info if present else nullptr
 */
inline BatmanInfer::BIITensorInfo *get_backing_tensor_info(BatmanInfer::graph::BITensor *tensor)
{
    return ((tensor == nullptr) || (tensor->handle() == nullptr)) ? nullptr : tensor->handle()->tensor().info();
}

/** Validates a ArgMinMax layer node
 *
 * @tparam ArgMinMax layer function type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ArgMinMaxLayer>
BIStatus validate_arg_min_max_layer(ArgMinMaxLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating ArgMinMaxLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output = get_backing_tensor_info(node.output(0));

    // Validate function
    return ArgMinMaxLayer::validate(input, node.axis(), output, node.reduction_operation());
}

/** Validates a Bounding Box Transform layer node
 *
 * @tparam BoundingBoxTransformLayer  Bounding Box Transform layer function type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename BoundingBoxTransformLayer>
BIStatus validate_bounding_box_transform_layer(BoundingBoxTransformLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating BoundingBoxTransformLayer node with ID : " << node.id() << " and Name: "
                                                                                        << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 2);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo      *input     = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo      *deltas    = get_backing_tensor_info(node.input(1));
    BatmanInfer::BIITensorInfo      *output    = get_backing_tensor_info(node.output(0));
    const BIBoundingBoxTransformInfo bbox_info = node.info();

    return BoundingBoxTransformLayer::validate(input, output, deltas, bbox_info);
}

/** Validates a Channel Shuffle layer node
 *
 * @tparam ChannelShuffleLayer  Channel Shuffle layer function type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ChannelShuffleLayer>
BIStatus validate_channel_shuffle_layer(ChannelShuffleLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating ChannelShuffle node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input      = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output     = get_backing_tensor_info(node.output(0));
    const unsigned int          num_groups = node.num_groups();

    return ChannelShuffleLayer::validate(input, output, num_groups);
}

/** Validates a Convolution layer node
 *
 * @tparam ConvolutionLayer          Default Convolution layer function type
 * @tparam DirectConvolutionLayer    Direct Convolution layer function type
 * @tparam GEMMConvolutionLayer      GEMM Convolution layer function type
 * @tparam WinogradConvolutionLayer  Winograd Convolution layer function type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ConvolutionLayer,
          typename DirectConvolutionLayer,
          typename GEMMConvolutionLayer,
          typename WinogradConvolutionLayer>
BIStatus validate_convolution_layer(ConvolutionLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating ConvolutionLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input   = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *weights = get_backing_tensor_info(node.input(1));
    BatmanInfer::BIITensorInfo *biases  = get_backing_tensor_info(node.input(2));
    BatmanInfer::BIITensorInfo *output  = get_backing_tensor_info(node.output(0));

    if (is_data_type_quantized_asymmetric(input->data_type()))
    {
        biases->set_data_type(BIDataType::S32);
    }

    const BIPadStrideInfo     conv_info      = node.convolution_info();
    const BIConvolutionMethod conv_algorithm = node.convolution_method();
    const bool                fast_math      = node.fast_math_hint() == BIFastMathHint::Enabled;
    const unsigned int        num_groups     = node.num_groups();

    // Validate function
    BIStatus status{};
    switch (conv_algorithm)
    {
        case BIConvolutionMethod::Direct:
            BI_COMPUTE_RETURN_ERROR_ON_MSG(num_groups != 1, "DirectConvolutionLayer does not support grouping!");
            status = DirectConvolutionLayer::validate(input, weights, biases, output, conv_info);
            break;
        case BIConvolutionMethod::GEMM:
            status = GEMMConvolutionLayer::validate(input, weights, biases, output, conv_info, BIWeightsInfo(),
                                                    Size2D(1, 1), BIActivationLayerInfo(), num_groups);
            break;
        case BIConvolutionMethod::Winograd:
            BI_COMPUTE_RETURN_ERROR_ON_MSG(num_groups != 1, "WinogradConvolutionLayer does not support grouping!");
            status = WinogradConvolutionLayer::validate(input, weights, biases, output, conv_info,
                                                        BIActivationLayerInfo(), fast_math);
            break;
        case BIConvolutionMethod::Default:
            status = ConvolutionLayer::validate(input, weights, biases, output, conv_info, BIWeightsInfo(), Size2D(1, 1),
                                                BIActivationLayerInfo(), fast_math, num_groups);
            break;
        default:
            BI_COMPUTE_RETURN_ERROR_MSG("Unsupported convolution method");
    }

    return status;
}

/** Validates a Depthwise Convolution layer node
 *
 * @tparam DepthwiseConvolutionLayer    Default Depthwise Convolution layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename DepthwiseConvolutionLayer>
BIStatus validate_depthwise_convolution_layer(DepthwiseConvolutionLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating DepthwiseConvolutionLayer node with ID : " << node.id() << " and Name: "
                                                                                        << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input   = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *weights = detail::get_backing_tensor_info(node.input(1));
    BatmanInfer::BIITensorInfo *biases  = get_backing_tensor_info(node.input(2));
    BatmanInfer::BIITensorInfo *output  = get_backing_tensor_info(node.output(0));

    const BIPadStrideInfo              conv_info        = node.convolution_info();
    const BIDepthwiseConvolutionMethod dwc_algorithm    = node.depthwise_convolution_method();
    const int                          depth_multiplier = node.depth_multiplier();

    // Validate function
    BIStatus status{};
    switch (dwc_algorithm)
    {
        case BIDepthwiseConvolutionMethod::Default:
        case BIDepthwiseConvolutionMethod::Optimized3x3:
            status = DepthwiseConvolutionLayer::validate(input, weights, biases, output, conv_info, depth_multiplier);
            break;
        default:
            BI_COMPUTE_RETURN_ERROR_MSG("Unsupported depthwise convolution method");
    }

    return status;
}

/** Validates a depth to space layer node
 *
 * @tparam DequantizationLayer Dequantize layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename DepthToSpaceLayer>
BIStatus validate_depth_to_space_layer(DepthToSpaceLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating DetectionOutputLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input  = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output = get_backing_tensor_info(node.output(0));

    return DepthToSpaceLayer::validate(input, output, node.block_shape());
}

/** Validates a dequantize layer node
 *
 * @tparam DequantizationLayer Dequantize layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename DequantizationLayer>
BIStatus validate_dequantization_layer(DequantizationLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating DetectionOutputLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input  = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output = get_backing_tensor_info(node.output(0));

    return DequantizationLayer::validate(input, output);
}

/** Validates a detection output layer node
 *
 * @tparam DetectionOutputLayer DetectionOutput layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename DetectionOutputLayer>
BIStatus validate_detection_output_layer(DetectionOutputLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating DetectionOutputLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo      *input0      = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo      *input1      = get_backing_tensor_info(node.input(1));
    BatmanInfer::BIITensorInfo      *input2      = get_backing_tensor_info(node.input(2));
    BatmanInfer::BIITensorInfo      *output      = get_backing_tensor_info(node.output(0));
    const BIDetectionOutputLayerInfo detect_info = node.detection_output_info();

    return DetectionOutputLayer::validate(input0, input1, input2, output, detect_info);
}

/** Validates a detection post process layer node
 *
 * @tparam DetectionPostProcessLayer DetectionOutput layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename DetectionPostProcessLayer>
BIStatus validate_detection_post_process_layer(DetectionPostProcessLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating DetectionPostProcessLayer node with ID : " << node.id() << " and Name: "
                                                                                        << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 4);

    // Extract IO and info
    BatmanInfer::BIITensorInfo           *input0      = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo           *input1      = get_backing_tensor_info(node.input(1));
    BatmanInfer::BIITensorInfo           *input2      = get_backing_tensor_info(node.input(2));
    BatmanInfer::BIITensorInfo           *output0     = get_backing_tensor_info(node.output(0));
    BatmanInfer::BIITensorInfo           *output1     = get_backing_tensor_info(node.output(1));
    BatmanInfer::BIITensorInfo           *output2     = get_backing_tensor_info(node.output(2));
    BatmanInfer::BIITensorInfo           *output3     = get_backing_tensor_info(node.output(3));
    const BIDetectionPostProcessLayerInfo detect_info = node.detection_post_process_info();

    return DetectionPostProcessLayer::validate(input0, input1, input2, output0, output1, output2, output3, detect_info);
}

/** Validates a Generate Proposals layer node
 *
 * @tparam GenerateProposalsLayer Generate Proposals layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename GenerateProposalsLayer>
BIStatus validate_generate_proposals_layer(GenerateProposalsLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating GenerateProposalsLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 3);

    // Extract IO and info
    BatmanInfer::BIITensorInfo   *scores              = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo   *deltas              = detail::get_backing_tensor_info(node.input(1));
    BatmanInfer::BIITensorInfo   *anchors             = detail::get_backing_tensor_info(node.input(2));
    BatmanInfer::BIITensorInfo   *proposals           = get_backing_tensor_info(node.output(0));
    BatmanInfer::BIITensorInfo   *scores_out          = get_backing_tensor_info(node.output(1));
    BatmanInfer::BIITensorInfo   *num_valid_proposals = get_backing_tensor_info(node.output(2));
    const BIGenerateProposalsInfo info                = node.info();

    return GenerateProposalsLayer::validate(scores, deltas, anchors, proposals, scores_out, num_valid_proposals, info);
}

/** Validates a L2Normalization layer node
 *
 * @tparam L2Normalization layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename L2NormalizeLayer>
BIStatus validate_l2_normalize_layer(L2NormalizeLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating L2NormalizeLayerNode node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input   = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output  = get_backing_tensor_info(node.output(0));
    int                       axis    = node.axis();
    float                     epsilon = node.epsilon();

    // Validate function
    return L2NormalizeLayer::validate(input, output, axis, epsilon);
}

/** Validates a NormalizePlanarYUV layer node
 *
 * @tparam NormalizePlanarYUVLayer layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename NormalizePlanarYUVLayer>
BIStatus validate_normalize_planar_yuv_layer(NormalizePlanarYUVLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating NormalizePlanarYUVLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 3);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *mean   = detail::get_backing_tensor_info(node.input(1));
    BatmanInfer::BIITensorInfo *std    = detail::get_backing_tensor_info(node.input(2));
    BatmanInfer::BIITensorInfo *output = get_backing_tensor_info(node.output(0));

    // Validate function
    return NormalizePlanarYUVLayer::validate(input, output, mean, std);
}

/** Validates a pad layer node
 *
 * @tparam PadLayer Pad layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename PadLayer>
BIStatus validate_pad_layer(PadLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating PadLayer node with ID : " << node.id() << " and Name: " << node.name()
                                                                       << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input   = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output  = get_backing_tensor_info(node.output(0));
    const PaddingList        &padding = node.padding();

    return PadLayer::validate(input, output, padding);
}

/** Validates a permute layer node
 *
 * @tparam PermuteLayer Permute layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename PermuteLayer>
BIStatus validate_permute_layer(PermuteLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating PermuteLayer node with ID : " << node.id() << " and Name: " << node.name()
                                                                           << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input  = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output = get_backing_tensor_info(node.output(0));
    const PermutationVector  &perm   = node.permutation_vector();

    return PermuteLayer::validate(input, output, perm);
}

/** Validates a PRelu layer node
 *
 * @tparam PReluLayer PRelu layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename PReluLayer>
BIStatus validate_prelu_layer(PReluLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating PRelu node with ID : " << node.id() << " and Name: " << node.name()
                                                                    << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 2);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input  = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *alpha  = get_backing_tensor_info(node.input(1));
    BatmanInfer::BIITensorInfo *output = get_backing_tensor_info(node.output(0));

    return PReluLayer::validate(input, alpha, output);
}

/** Validates a priorbox layer node
 *
 * @tparam PriorBoxLayer PriorBox layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename PriorBoxLayer>
BIStatus validate_priorbox_layer(PriorBoxLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating PriorBoxLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 2);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input0     = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *input1     = get_backing_tensor_info(node.input(1));
    BatmanInfer::BIITensorInfo *output     = get_backing_tensor_info(node.output(0));
    const BIPriorBoxLayerInfo   prior_info = node.priorbox_info();

    return PriorBoxLayer::validate(input0, input1, output, prior_info);
}

/** Validates a Quantization layer node
 *
 * @tparam QuantizationLayer Quantization layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename QuantizationLayer>
BIStatus validate_quantization_layer(QuantizationLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating QuantizationLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    BatmanInfer::BIITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output = get_backing_tensor_info(node.output(0));

    // Validate function
    return QuantizationLayer::validate(input, output);
}

/** Validates a Reduction operation layer node
 *
 * @tparam ReductionLayer Reduction layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ReductionLayer>
BIStatus validate_reduction_operation_layer(ReductionLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating ReductionLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);

    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    BatmanInfer::BIITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output = get_backing_tensor_info(node.output(0));

    // Validate function
    return ReductionLayer::validate(input, output, node.axis(), node.op(), node.keep_dims());
}

/** Validates a Reorg layer node
 *
 * @tparam ReorgLayer Reorg layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ReorgLayer>
BIStatus validate_reorg_layer(ReorgLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating ReorgLayer node with ID : " << node.id() << " and Name: " << node.name()
                                                                         << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    BatmanInfer::BIITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output = get_backing_tensor_info(node.output(0));

    // Validate function
    return ReorgLayer::validate(input, output, node.stride());
}

/** Validates a Reshape layer node
 *
 * @tparam ReshapeLayer Reshape layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ReshapeLayer>
BIStatus validate_reshape_layer(ReshapeLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating ReshapeLayer node with ID : " << node.id() << " and Name: " << node.name()
                                                                           << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    BatmanInfer::BIITensorInfo *input  = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output = detail::get_backing_tensor_info(node.output(0));

    // Validate function
    return ReshapeLayer::validate(input, output);
}

/** Validates a ROI Align layer node
 *
 * @tparam ROIAlignLayer ROIAlign layer type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename ROIAlignLayer>
BIStatus validate_roi_align_layer(ROIAlignLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE(
        "Validating ROIAlignLayer node with ID : " << node.id() << " and Name: " << node.name() << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 2);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    BatmanInfer::BIITensorInfo  *input     = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo  *rois      = detail::get_backing_tensor_info(node.input(1));
    BatmanInfer::BIITensorInfo  *output    = detail::get_backing_tensor_info(node.output(0));
    const BIROIPoolingLayerInfo &pool_info = node.pooling_info();

    // Validate function
    return ROIAlignLayer::validate(input, rois, output, pool_info);
}

/** Validates a Slice layer node
 *
 * @tparam SliceLayer Slice layer function type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename SliceLayer>
BIStatus validate_slice_layer(SliceLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating Slice node with ID : " << node.id() << " and Name: " << node.name()
                                                                    << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo *input  = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo *output = get_backing_tensor_info(node.output(0));
    const BICoordinates         starts = node.starts();
    const BICoordinates         ends   = node.ends();

    return SliceLayer::validate(input, output, starts, ends);
}

/** Validates a Strided Slice layer node
 *
 * @tparam StridedSliceLayer Strided Slice layer function type
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename StridedSliceLayer>
BIStatus validate_strided_slice_layer(StridedSliceLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating StridedSlice node with ID : " << node.id() << " and Name: " << node.name()
                                                                           << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract IO and info
    BatmanInfer::BIITensorInfo   *input   = get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo   *output  = get_backing_tensor_info(node.output(0));
    const BICoordinates           starts  = node.starts();
    const BICoordinates           ends    = node.ends();
    const BiStrides               strides = node.strides();
    const BIStridedSliceLayerInfo info    = node.strided_slice_info();

    return StridedSliceLayer::validate(input, output, starts, ends, strides, info.begin_mask(), info.end_mask(),
                                       info.shrink_axis_mask());
}

/** Validates a element-wise layer node
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename EltwiseLayerFunctions>
BIStatus validate_eltwise_Layer(EltwiseLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating EltwiseLayer node with ID : " << node.id() << " and Name: " << node.name()
                                                                           << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 2);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    const BatmanInfer::BIITensorInfo *input1         = detail::get_backing_tensor_info(node.input(0));
    const BatmanInfer::BIITensorInfo *input2         = detail::get_backing_tensor_info(node.input(1));
    const BatmanInfer::BIITensorInfo *output         = get_backing_tensor_info(node.output(0));
    const BIEltwiseOperation          eltwise_op     = node.eltwise_operation();
    const BIConvertPolicy             convert_policy = node.convert_policy();
    const BIRoundingPolicy            round_policy   = node.rounding_policy();
    const BIActivationLayerInfo       act_info       = node.fused_activation();
    const BIQuantizationInfo          quant_info     = node.output_quant_info();

    // Validate function
    if (eltwise_op == BIEltwiseOperation::Add)
    {
        return EltwiseLayerFunctions::ArithmeticAddition::validate(input1, input2, output, convert_policy, act_info);
    }
    else if (eltwise_op == BIEltwiseOperation::Sub)
    {
        return EltwiseLayerFunctions::ArithmeticSubtraction::validate(input1, input2, output, convert_policy, act_info);
    }
    else if (eltwise_op == BIEltwiseOperation::Mul)
    {
        return EltwiseLayerFunctions::PixelWiseMultiplication::validate(input1, input2, output, 1.0f, convert_policy,
                                                                        round_policy, act_info);
    }
    else if (eltwise_op == BIEltwiseOperation::Max)
    {
        return EltwiseLayerFunctions::ElementwiseMax::validate(input1, input2, output, act_info);
    }
    else if (eltwise_op == BIEltwiseOperation::Div)
    {
        return EltwiseLayerFunctions::ArithmeticDivision::validate(input1, input2, output, act_info);
    }
    else
    {
        BI_COMPUTE_ERROR("Unsupported element-wise operation!");
    }
    return BIStatus{};
}
/** Validates a unary element-wise layer node
 *
 * @param[in] node Node to validate
 *
 * @return Status
 */
template <typename UnaryEltwiseLayerFunctions>
BIStatus validate_unary_eltwise_layer(UnaryEltwiseLayerNode &node)
{
    BI_COMPUTE_LOG_GRAPH_VERBOSE("Validating EltwiseLayer node with ID : " << node.id() << " and Name: " << node.name()
                                                                           << std::endl);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_inputs() != 1);
    BI_COMPUTE_RETURN_ERROR_ON(node.num_outputs() != 1);

    // Extract input and output
    BatmanInfer::BIITensorInfo   *input      = detail::get_backing_tensor_info(node.input(0));
    BatmanInfer::BIITensorInfo   *output     = get_backing_tensor_info(node.output(0));
    const BIUnaryEltwiseOperation eltwise_op = node.eltwise_descriptor().op;

    // Validate function
    if (eltwise_op == BIUnaryEltwiseOperation::Exp)
    {
        return UnaryEltwiseLayerFunctions::ExpLayer::validate(input, output);
    }
    else
    {
        BI_COMPUTE_ERROR("Unsupported unary element-wise operation!");
    }

    return BIStatus{};
}

} // namespace detail

} // namespace backends

} // namespace graph

} // namespace BatmanInfer
