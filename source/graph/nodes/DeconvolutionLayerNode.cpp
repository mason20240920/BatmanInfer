//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/DeconvolutionLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    DeconvolutionLayerNode::DeconvolutionLayerNode(const descriptors::DeconvolutionLayerDescriptor &descriptor)
    : descriptor(std::move(descriptor))
    {
        _input_edges.resize(3, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BIPadStrideInfo DeconvolutionLayerNode::deconvolution_info() const
    {
        return descriptor.info;
    }

    BITensorDescriptor DeconvolutionLayerNode::compute_output_descriptor(const BITensorDescriptor &input_descriptor,
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
            deconvolution_output_dimensions(input_width, input_height, kernel_width, kernel_height, info);

        const BIDataLayout data_layout       = input_descriptor.layout;
        BITensorDescriptor output_descriptor = input_descriptor;
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::WIDTH), output_width);
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::HEIGHT), output_height);
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::CHANNEL),
                                    weights_descriptor.shape[3]);

        return output_descriptor;
    }

    bool DeconvolutionLayerNode::forward_descriptors()
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

    BITensorDescriptor DeconvolutionLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        const BITensor *src     = input(0);
        const BITensor *weights = input(1);

        BI_COMPUTE_ERROR_ON(src == nullptr || weights == nullptr);

        BITensorDescriptor output_info = compute_output_descriptor(src->desc(), weights->desc(), descriptor.info);

        if (!descriptor.out_quant_info.empty())
        {
            output_info.set_quantization_info(descriptor.out_quant_info);
        }

        return output_info;
    }

    BINodeType DeconvolutionLayerNode::type() const
    {
        return BINodeType::DeconvolutionLayer;
    }

    void DeconvolutionLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
