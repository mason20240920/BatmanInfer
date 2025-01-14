//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/PrintLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    PrintLayerNode::PrintLayerNode(std::ostream                                 &stream,
                                   const BIIOFormatInfo                         &format_info,
                                   const std::function<BIITensor *(BIITensor *)> transform)
    : _stream(stream), _format_info(format_info), _transform(transform)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    std::ostream &PrintLayerNode::stream() const
    {
        return _stream;
    }

    const BIIOFormatInfo PrintLayerNode::format_info() const
    {
        return _format_info;
    }

    const std::function<BIITensor *(BIITensor *)> PrintLayerNode::transform() const
    {
        return _transform;
    }

    bool PrintLayerNode::forward_descriptors()
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

    BITensorDescriptor PrintLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor output_desc = src->desc();
        return output_desc;
    }

    BINodeType PrintLayerNode::type() const
    {
        return BINodeType::PrintLayer;
    }

    void PrintLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
