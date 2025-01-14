//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/ReorgLayerNode.h"

#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    ReorgLayerNode::ReorgLayerNode(int stride) : _stride(stride)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    int ReorgLayerNode::stride() const
    {
        return _stride;
    }

    BITensorDescriptor ReorgLayerNode::compute_output_descriptor(const BITensorDescriptor &input_descriptor, int stride)
    {
        const unsigned int input_width   = get_dimension_size(input_descriptor, BIDataLayoutDimension::WIDTH);
        const unsigned int input_height  = get_dimension_size(input_descriptor, BIDataLayoutDimension::HEIGHT);
        const unsigned int input_channel = get_dimension_size(input_descriptor, BIDataLayoutDimension::CHANNEL);

        BI_COMPUTE_ERROR_ON(stride <= 0);
        BI_COMPUTE_ERROR_ON_MSG((input_width % stride != 0), "The width of the input tensor must be a multiple of stride");
        BI_COMPUTE_ERROR_ON_MSG((input_height % stride != 0),
                                 "The height of the input tensor must be a multiple of stride");

        const BIDataLayout data_layout       = input_descriptor.layout;
        BITensorDescriptor output_descriptor = input_descriptor;
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::WIDTH), input_width / stride);
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::HEIGHT), input_height / stride);
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::CHANNEL),
                                    input_channel * stride * stride);

        return output_descriptor;
    }

    bool ReorgLayerNode::forward_descriptors()
    {
        if ((input_id(0) != NullTensorID) && (output_id(0) != NullTensorID))
        {
            BITensor *dst = output(0);
            BI_COMPUTE_ERROR_ON(dst == nullptr);
            dst->desc() = configure_output(0);
            return true;
        }
        return false;
    }

    BITensorDescriptor ReorgLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        return compute_output_descriptor(src->desc(), _stride);
    }

    BINodeType ReorgLayerNode::type() const
    {
        return BINodeType::ReorgLayer;
    }

    void ReorgLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
