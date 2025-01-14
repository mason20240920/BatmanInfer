//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/NormalizePlanarYUVLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    NormalizePlanarYUVLayerNode::NormalizePlanarYUVLayerNode()
    {
        _input_edges.resize(3, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    bool NormalizePlanarYUVLayerNode::forward_descriptors()
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

    BITensorDescriptor NormalizePlanarYUVLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        return src->desc();
    }

    BINodeType NormalizePlanarYUVLayerNode::type() const
    {
        return BINodeType::NormalizePlanarYUVLayer;
    }

    void NormalizePlanarYUVLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
