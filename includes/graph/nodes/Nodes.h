//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_NODES_H
#define BATMANINFER_GRAPH_NODES_H

#include "graph/nodes/ActivationLayerNode.h"
#include "graph/nodes/ArgMinMaxLayerNode.h"
#include "graph/nodes/BatchNormalizationLayerNode.h"
#include "graph/nodes/BoundingBoxTransformLayerNode.h"
#include "graph/nodes/ChannelShuffleLayerNode.h"
#include "graph/nodes/ConcatenateLayerNode.h"
#include "graph/nodes/ConstNode.h"
#include "graph/nodes/ConvolutionLayerNode.h"
#include "graph/nodes/DeconvolutionLayerNode.h"
#include "graph/nodes/DepthToSpaceLayerNode.h"
#include "graph/nodes/DepthwiseConvolutionLayerNode.h"
#include "graph/nodes/DequantizationLayerNode.h"
#include "graph/nodes/DetectionOutputLayerNode.h"
#include "graph/nodes/DetectionPostProcessLayerNode.h"
#include "graph/nodes/DummyNode.h"
#include "graph/nodes/EltwiseLayerNode.h"
#include "graph/nodes/FlattenLayerNode.h"
#include "graph/nodes/FullyConnectedLayerNode.h"
#include "graph/nodes/FusedConvolutionBatchNormalizationNode.h"
#include "graph/nodes/FusedDepthwiseConvolutionBatchNormalizationNode.h"
#include "graph/nodes/GenerateProposalsLayerNode.h"
#include "graph/nodes/InputNode.h"
#include "graph/nodes/L2NormalizeLayerNode.h"
#include "graph/nodes/NormalizationLayerNode.h"
#include "graph/nodes/NormalizePlanarYUVLayerNode.h"
#include "graph/nodes/OutputNode.h"
#include "graph/nodes/PadLayerNode.h"
#include "graph/nodes/PermuteLayerNode.h"
#include "graph/nodes/PoolingLayerNode.h"
#include "graph/nodes/PReluLayerNode.h"
#include "graph/nodes/PrintLayerNode.h"
#include "graph/nodes/PriorBoxLayerNode.h"
#include "graph/nodes/QuantizationLayerNode.h"
#include "graph/nodes/ReductionLayerNode.h"
#include "graph/nodes/ReorgLayerNode.h"
#include "graph/nodes/ReshapeLayerNode.h"
#include "graph/nodes/ResizeLayerNode.h"
#include "graph/nodes/ROIAlignLayerNode.h"
#include "graph/nodes/SliceLayerNode.h"
#include "graph/nodes/SoftmaxLayerNode.h"
#include "graph/nodes/SplitLayerNode.h"
#include "graph/nodes/StackLayerNode.h"
#include "graph/nodes/StridedSliceLayerNode.h"

#endif //BATMANINFER_GRAPH_NODES_H