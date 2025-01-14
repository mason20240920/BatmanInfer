//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/PermuteLayerNode.h"

#include "data/core/bi_helpers.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    PermuteLayerNode::PermuteLayerNode(PermutationVector perm, BIDataLayout layout) : _perm(perm), _layout(layout)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    const PermutationVector &PermuteLayerNode::permutation_vector() const
    {
        return _perm;
    }

    bool PermuteLayerNode::forward_descriptors()
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

    BITensorDescriptor PermuteLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor output_desc = src->desc();
        permute(output_desc.shape, _perm);
        if (_layout != BIDataLayout::UNKNOWN)
        {
            output_desc.layout = _layout;
        }

        return output_desc;
    }

    BINodeType PermuteLayerNode::type() const
    {
        return BINodeType::PermuteLayer;
    }

    void PermuteLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
