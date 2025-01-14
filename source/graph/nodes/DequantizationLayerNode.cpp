//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/DequantizationLayerNode.h"

#include "data/core/bi_error.h"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_tensor.h"

namespace BatmanInfer {

namespace graph {

    DequantizationLayerNode::DequantizationLayerNode()
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    bool DequantizationLayerNode::forward_descriptors()
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

    BITensorDescriptor DequantizationLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor output_desc = src->desc();
        output_desc.data_type          = BIDataType::F32;

        return output_desc;
    }

    BINodeType DequantizationLayerNode::type() const
    {
        return BINodeType::DequantizationLayer;
    }

    void DequantizationLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
