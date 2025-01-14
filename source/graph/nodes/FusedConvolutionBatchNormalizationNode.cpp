//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/FusedConvolutionBatchNormalizationNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    FusedConvolutionBatchNormalizationNode::FusedConvolutionBatchNormalizationNode(
        float                 epsilon,
        BIPadStrideInfo       info,
        unsigned int          num_groups,
        BIConvolutionMethod   method,
        BIFastMathHint        fast_math_hint,
        BIActivationLayerInfo fused_activation)
    : _epsilon(epsilon),
      _info(std::move(info)),
      _num_groups(num_groups),
      _method(method),
      _fast_math_hint(fast_math_hint),
      _fused_activation(fused_activation)
    {
        _input_edges.resize(7, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    void FusedConvolutionBatchNormalizationNode::set_convolution_method(BIConvolutionMethod method)
    {
        _method = method;
    }

    float FusedConvolutionBatchNormalizationNode::epsilon() const
    {
        return _epsilon;
    }

    BIConvolutionMethod FusedConvolutionBatchNormalizationNode::convolution_method() const
    {
        return _method;
    }

    void FusedConvolutionBatchNormalizationNode::set_fast_math_hint(BIFastMathHint hint)
    {
        _fast_math_hint = hint;
    }

    BIFastMathHint FusedConvolutionBatchNormalizationNode::fast_math_hint() const
    {
        return _fast_math_hint;
    }

    BIPadStrideInfo FusedConvolutionBatchNormalizationNode::convolution_info() const
    {
        return _info;
    }

    unsigned int FusedConvolutionBatchNormalizationNode::num_groups() const
    {
        return _num_groups;
    }

    BIActivationLayerInfo FusedConvolutionBatchNormalizationNode::fused_activation() const
    {
        return _fused_activation;
    }

    void FusedConvolutionBatchNormalizationNode::set_fused_activation(BIActivationLayerInfo fused_activation)
    {
        _fused_activation = fused_activation;
    }

    BITensorDescriptor FusedConvolutionBatchNormalizationNode::compute_output_descriptor(
        const BITensorDescriptor &input_descriptor,
        const BITensorDescriptor &weights_descriptor,
        const BIPadStrideInfo    &info)
    {
        unsigned int output_width  = 0;
        unsigned int output_height = 0;

        const unsigned int input_width   = get_dimension_size(input_descriptor, BIDataLayoutDimension::WIDTH);
        const unsigned int input_height  = get_dimension_size(input_descriptor, BIDataLayoutDimension::HEIGHT);
        const unsigned int kernel_width  = get_dimension_size(weights_descriptor, BIDataLayoutDimension::WIDTH);
        const unsigned int kernel_height = get_dimension_size(weights_descriptor, BIDataLayoutDimension::HEIGHT);

        std::tie(output_width, output_height) =
            scaled_dimensions(input_width, input_height, kernel_width, kernel_height, info);

        const BIDataLayout data_layout       = input_descriptor.layout;
        BITensorDescriptor output_descriptor = input_descriptor;
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::WIDTH), output_width);
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::HEIGHT), output_height);
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::CHANNEL),
                                    weights_descriptor.shape[3]);

        return output_descriptor;
    }

    bool FusedConvolutionBatchNormalizationNode::forward_descriptors()
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

    BITensorDescriptor FusedConvolutionBatchNormalizationNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        const BITensor *src     = input(0);
        const BITensor *weights = input(1);

        BI_COMPUTE_ERROR_ON(src == nullptr || weights == nullptr);

        BITensorDescriptor output_info = compute_output_descriptor(src->desc(), weights->desc(), _info);

        return output_info;
    }

    BINodeType FusedConvolutionBatchNormalizationNode::type() const
    {
        return FusedConvolutionBatchNormalizationNode::node_type;
    }

    void FusedConvolutionBatchNormalizationNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
