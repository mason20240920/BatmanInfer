//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/GenerateProposalsLayerNode.h"

#include "data/core/bi_helpers.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    GenerateProposalsLayerNode::GenerateProposalsLayerNode(BIGenerateProposalsInfo &info) : _info(info)
    {
        _input_edges.resize(3, EmptyEdgeID);
        _outputs.resize(3, NullTensorID);
    }

    const BIGenerateProposalsInfo &GenerateProposalsLayerNode::info() const
    {
        return _info;
    }

    bool GenerateProposalsLayerNode::forward_descriptors()
    {
        if ((input_id(0) != NullTensorID) && (input_id(1) != NullTensorID) && (input_id(2) != NullTensorID) &&
            (output_id(0) != NullTensorID) && (output_id(1) != NullTensorID) && (output_id(2) != NullTensorID))
        {
            for (unsigned int i = 0; i < 3; ++i)
            {
                BITensor *dst = output(i);
                BI_COMPUTE_ERROR_ON(dst == nullptr);
                dst->desc() = configure_output(i);
            }
            return true;
        }
        return false;
    }

    BITensorDescriptor GenerateProposalsLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_ERROR_ON(idx > 3);

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);
        BITensorDescriptor output_desc = src->desc();

        switch (idx)
        {
            case 0:
                // Configure proposals output
                output_desc.shape = BITensorShape(5, src->desc().shape.total_size());
                break;
            case 1:
                // Configure scores_out output
                output_desc.shape = BITensorShape(src->desc().shape.total_size());
                break;
            case 2:
                // Configure num_valid_proposals
                output_desc.shape     = BITensorShape(1);
                output_desc.data_type = BIDataType::U32;
                break;
            default:
                BI_COMPUTE_ERROR("Unsupported output index");
        }
        return output_desc;
    }

    BINodeType GenerateProposalsLayerNode::type() const
    {
        return BINodeType::GenerateProposalsLayer;
    }

    void GenerateProposalsLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
