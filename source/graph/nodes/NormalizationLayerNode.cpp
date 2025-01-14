//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/NormalizationLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    NormalizationLayerNode::NormalizationLayerNode(BINormalizationLayerInfo norm_info) : _info(norm_info)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BINormalizationLayerInfo NormalizationLayerNode::normalization_info() const
    {
        return _info;
    }

    bool NormalizationLayerNode::forward_descriptors()
    {
        if (input_id(0) != NullTensorID && (output_id(0) != NullTensorID))
        {
            BITensor *dst = output(0);
            BI_COMPUTE_ERROR_ON(dst == nullptr);
            dst->desc() = configure_output(0);
            return true;
        }
        return false;
    }

    BITensorDescriptor NormalizationLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        return src->desc();
    }

    BINodeType NormalizationLayerNode::type() const
    {
        return BINodeType::NormalizationLayer;
    }

    void NormalizationLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
