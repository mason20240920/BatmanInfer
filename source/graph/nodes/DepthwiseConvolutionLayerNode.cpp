//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/DepthwiseConvolutionLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    DepthwiseConvolutionLayerNode::DepthwiseConvolutionLayerNode(BIPadStrideInfo              info,
                                                                 int                          depth_multiplier,
                                                                 BIDepthwiseConvolutionMethod method,
                                                                 BIQuantizationInfo           out_quant_info)
    : _info(std::move(info)),
      _depth_multiplier(depth_multiplier),
      _method(method),
      _out_quant_info(std::move(out_quant_info)),
      _fused_activation()
    {
        _input_edges.resize(3, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    int DepthwiseConvolutionLayerNode::depth_multiplier() const
    {
        return _depth_multiplier;
    }

    void DepthwiseConvolutionLayerNode::set_depthwise_convolution_method(BIDepthwiseConvolutionMethod method)
    {
        _method = method;
    }

    BIDepthwiseConvolutionMethod DepthwiseConvolutionLayerNode::depthwise_convolution_method() const
    {
        return _method;
    }

    BIPadStrideInfo DepthwiseConvolutionLayerNode::convolution_info() const
    {
        return _info;
    }

    BIActivationLayerInfo DepthwiseConvolutionLayerNode::fused_activation() const
    {
        return _fused_activation;
    }

    void DepthwiseConvolutionLayerNode::set_fused_activation(BIActivationLayerInfo fused_activation)
    {
        _fused_activation = fused_activation;
    }

    void DepthwiseConvolutionLayerNode::set_convolution_info(BIPadStrideInfo info)
    {
        _info = info;
    }

    BITensorDescriptor DepthwiseConvolutionLayerNode::compute_output_descriptor(
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

        const BIDataLayout data_layout       = input_descriptor.layout;
        BITensorDescriptor output_descriptor = input_descriptor;
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::WIDTH), output_width);
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::HEIGHT), output_height);
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::CHANNEL),
                                    input_channels * depth_multiplier);

        return output_descriptor;
    }

    bool DepthwiseConvolutionLayerNode::forward_descriptors()
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

    BITensorDescriptor DepthwiseConvolutionLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        const BITensor *src     = input(0);
        const BITensor *weights = input(1);

        BI_COMPUTE_ERROR_ON(src == nullptr || weights == nullptr);

        BITensorDescriptor output_info = compute_output_descriptor(src->desc(), weights->desc(), _info, _depth_multiplier);
        if (!_out_quant_info.empty())
        {
            output_info.quant_info = _out_quant_info;
        }

        return output_info;
    }

    BINodeType DepthwiseConvolutionLayerNode::type() const
    {
        return DepthwiseConvolutionLayerNode::node_type;
    }

    void DepthwiseConvolutionLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
