//
// Created by holynova on 2025/1/13.
//

#include "graph/nodes/BoundingBoxTransformLayerNode.h"

#include "data/core/bi_helpers.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    BoundingBoxTransformLayerNode::BoundingBoxTransformLayerNode(BIBoundingBoxTransformInfo &info) : _bbox_info(info)
    {
        _input_edges.resize(2, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    const BIBoundingBoxTransformInfo &BoundingBoxTransformLayerNode::info() const
    {
        return _bbox_info;
    }

    bool BoundingBoxTransformLayerNode::forward_descriptors()
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

    BITensorDescriptor BoundingBoxTransformLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *deltas = input(1);
        BI_COMPUTE_ERROR_ON(deltas == nullptr);

        BITensorDescriptor output_desc = deltas->desc();
        return output_desc;
    }

    BINodeType BoundingBoxTransformLayerNode::type() const
    {
        return BINodeType::BoundingBoxTransformLayer;
    }

    void BoundingBoxTransformLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
