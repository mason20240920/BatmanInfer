//
// Created by holynova on 2025/1/13.
//

#include "graph/nodes/ChannelShuffleLayerNode.h"

#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    ChannelShuffleLayerNode::ChannelShuffleLayerNode(unsigned int num_groups) : _num_groups(num_groups)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    unsigned int ChannelShuffleLayerNode::num_groups() const
    {
        return _num_groups;
    }

    bool ChannelShuffleLayerNode::forward_descriptors()
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

    BITensorDescriptor ChannelShuffleLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        return src->desc();
    }

    BINodeType ChannelShuffleLayerNode::type() const
    {
        return BINodeType::ChannelShuffleLayer;
    }

    void ChannelShuffleLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
