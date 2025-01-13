//
// Created by holynova on 2025/1/10.
//

#include "graph/nodes/ActivationLayerNode.h"

#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    ActivationLayerNode::ActivationLayerNode(BIActivationLayerInfo info, BIQuantizationInfo out_quant_info)
    : _info(info), _out_quant_info(std::move(out_quant_info))
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BIActivationLayerInfo ActivationLayerNode::activation_info() const
    {
        return _info;
    }

    bool ActivationLayerNode::forward_descriptors()
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

    BITensorDescriptor ActivationLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor output_info = src->desc();
        if (!_out_quant_info.empty())
        {
            output_info.quant_info = _out_quant_info;
        }

        return output_info;
    }

    BINodeType ActivationLayerNode::type() const
    {
        return ActivationLayerNode::node_type;
    }

    void ActivationLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
