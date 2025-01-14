//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/PadLayerNode.h"

#include "data/core/bi_helpers.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    PadLayerNode::PadLayerNode(const PaddingList &padding, BIPixelValue pad_value)
        : _padding(padding), _pad_value(pad_value)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    const PaddingList &PadLayerNode::padding() const
    {
        return _padding;
    }

    BIPixelValue PadLayerNode::pad_value() const
    {
        return _pad_value;
    }

    bool PadLayerNode::forward_descriptors()
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

    BITensorDescriptor PadLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor  output_desc = src->desc();
        const BITensorShape input_shape = src->desc().shape;
        for (size_t dim = 0; dim < _padding.size(); ++dim)
        {
            output_desc.shape.set(dim, _padding[dim].first + input_shape[dim] + _padding[dim].second);
        }

        return output_desc;
    }

    BINodeType PadLayerNode::type() const
    {
        return BINodeType::PadLayer;
    }

    void PadLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
