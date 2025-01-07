//
// Created by holynova on 2025/1/2.
//

#ifndef BATMANINFER_GRAPH_NODESFWD_H
#define BATMANINFER_GRAPH_NODESFWD_H

namespace BatmanInfer {

namespace graph {

    // Forward declarations
    class BIINode;
    class ActivationLayerNode;
    class ArgMinMaxLayerNode;
    class BatchNormalizationLayerNode;
    class BoundingBoxTransformLayerNode;
    class ChannelShuffleLayerNode;
    class ConcatenateLayerNode;
    class ConstNode;
    class ConvolutionLayerNode;
    class DeconvolutionLayerNode;
    class DepthToSpaceLayerNode;
    class DepthwiseConvolutionLayerNode;
    class DequantizationLayerNode;
    class DetectionOutputLayerNode;
    class DetectionPostProcessLayerNode;
    class DummyNode;
    class EltwiseLayerNode;
    class FlattenLayerNode;
    class FullyConnectedLayerNode;
    class FusedConvolutionBatchNormalizationNode;
    class FusedDepthwiseConvolutionBatchNormalizationNode;
    class GenerateProposalsLayerNode;
    class InputNode;
    class L2NormalizeLayerNode;
    class NormalizationLayerNode;
    class NormalizePlanarYUVLayerNode;
    class OutputNode;
    class PadLayerNode;
    class PermuteLayerNode;
    class PoolingLayerNode;
    class PReluLayerNode;
    class PrintLayerNode;
    class PriorBoxLayerNode;
    class QuantizationLayerNode;
    class ReductionLayerNode;
    class ReorgLayerNode;
    class ReshapeLayerNode;
    class ResizeLayerNode;
    class ROIAlignLayerNode;
    class SoftmaxLayerNode;
    class SliceLayerNode;
    class SplitLayerNode;
    class StackLayerNode;
    class StridedSliceLayerNode;

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_NODESFWD_H
