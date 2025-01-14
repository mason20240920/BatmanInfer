//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/L2NormalizeLayerNode.h"

#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    L2NormalizeLayerNode::L2NormalizeLayerNode() : L2NormalizeLayerNode(0, 1e-12f)
    {
    }

    L2NormalizeLayerNode::L2NormalizeLayerNode(int axis) : L2NormalizeLayerNode(axis, 1e-12f)
    {
    }

    L2NormalizeLayerNode::L2NormalizeLayerNode(int axis, float epsilon) : _axis(axis), _epsilon(epsilon)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    bool L2NormalizeLayerNode::forward_descriptors()
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

    BITensorDescriptor L2NormalizeLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor output_desc = src->desc();

        return output_desc;
    }

    int L2NormalizeLayerNode::axis() const
    {
        return _axis;
    }

    float L2NormalizeLayerNode::epsilon() const
    {
        return _epsilon;
    }

    BINodeType L2NormalizeLayerNode::type() const
    {
        return BINodeType::L2NormalizeLayer;
    }

    void L2NormalizeLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
