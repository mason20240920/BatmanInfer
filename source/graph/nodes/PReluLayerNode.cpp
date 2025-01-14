//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/PReluLayerNode.h"

#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    PReluLayerNode::PReluLayerNode()
    {
        _input_edges.resize(2, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    bool PReluLayerNode::forward_descriptors()
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

    BITensorDescriptor PReluLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        return src->desc();
    }

    BINodeType PReluLayerNode::type() const
    {
        return BINodeType::PReluLayer;
    }

    void PReluLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
