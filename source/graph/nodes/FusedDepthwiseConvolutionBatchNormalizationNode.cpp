//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/FusedDepthwiseConvolutionBatchNormalizationNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    FusedDepthwiseConvolutionBatchNormalizationNode::FusedDepthwiseConvolutionBatchNormalizationNode(
        float                        epsilon,
        BIPadStrideInfo              info,
        unsigned int                 depth_multiplier,
        BIDepthwiseConvolutionMethod method,
        BIActivationLayerInfo        fused_activation)
    : _epsilon(epsilon),
      _info(std::move(info)),
      _depth_multiplier(depth_multiplier),
      _method(method),
      _fused_activation(fused_activation)
    {
        _input_edges.resize(7, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    void FusedDepthwiseConvolutionBatchNormalizationNode::set_depthwise_convolution_method(
        BIDepthwiseConvolutionMethod method)
    {
        _method = method;
    }

    BIDepthwiseConvolutionMethod FusedDepthwiseConvolutionBatchNormalizationNode::depthwise_convolution_method() const
    {
        return _method;
    }

    float FusedDepthwiseConvolutionBatchNormalizationNode::epsilon() const
    {
        return _epsilon;
    }

    BIPadStrideInfo FusedDepthwiseConvolutionBatchNormalizationNode::convolution_info() const
    {
        return _info;
    }

    unsigned int FusedDepthwiseConvolutionBatchNormalizationNode::depth_multiplier() const
    {
        return _depth_multiplier;
    }

    BIActivationLayerInfo FusedDepthwiseConvolutionBatchNormalizationNode::fused_activation() const
    {
        return _fused_activation;
    }

    void FusedDepthwiseConvolutionBatchNormalizationNode::set_fused_activation(BIActivationLayerInfo fused_activation)
    {
        _fused_activation = fused_activation;
    }

    BITensorDescriptor FusedDepthwiseConvolutionBatchNormalizationNode::compute_output_descriptor(
        const BITensorDescriptor &input_descriptor,
        const BITensorDescriptor &weights_descriptor,
        const BIPadStrideInfo    &info,
        int                       depth_multiplier)
    {
        unsigned int output_width  = 0;
        unsigned int output_height = 0;

        const unsigned int input_width    = get_dimension_size(input_descriptor, BIDataLayoutDimension::WIDTH);
        const unsigned int input_height   = get_dimension_size(input_descriptor, BIDataLayoutDimension::HEIGHT);
        const unsigned int input_channels = get_dimension_size(input_descriptor, BIDataLayoutDimension::CHANNEL);
        const unsigned int kernel_width   = get_dimension_size(weights_descriptor, BIDataLayoutDimension::WIDTH);
        const unsigned int kernel_height  = get_dimension_size(weights_descriptor, BIDataLayoutDimension::HEIGHT);

        std::tie(output_width, output_height) =
            scaled_dimensions(input_width, input_height, kernel_width, kernel_height, info);

        BITensorDescriptor output_descriptor = input_descriptor;
        output_descriptor.shape.set(get_dimension_idx(output_descriptor.layout, BIDataLayoutDimension::WIDTH),
                                    output_width);
        output_descriptor.shape.set(get_dimension_idx(output_descriptor.layout, BIDataLayoutDimension::HEIGHT),
                                    output_height);
        output_descriptor.shape.set(get_dimension_idx(output_descriptor.layout, BIDataLayoutDimension::CHANNEL),
                                    input_channels * depth_multiplier);

        return output_descriptor;
    }

    bool FusedDepthwiseConvolutionBatchNormalizationNode::forward_descriptors()
    {
        if ((input_id(0) != NullTensorID) && (input_id(1) != NullTensorID) && (output_id(0) != NullTensorID))
        {
            BITensor *dst = output(0);
            BI_COMPUTE_ERROR_ON(dst == nullptr);
            dst->desc() = configure_output(0);
            return true;
        }
        return false;
    }

    BITensorDescriptor FusedDepthwiseConvolutionBatchNormalizationNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        const BITensor *src     = input(0);
        const BITensor *weights = input(1);

        BI_COMPUTE_ERROR_ON(src == nullptr || weights == nullptr);

        BITensorDescriptor output_info = compute_output_descriptor(src->desc(), weights->desc(), _info, _depth_multiplier);

        return output_info;
    }

    BINodeType FusedDepthwiseConvolutionBatchNormalizationNode::type() const
    {
        return FusedDepthwiseConvolutionBatchNormalizationNode::node_type;
    }

    void FusedDepthwiseConvolutionBatchNormalizationNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
