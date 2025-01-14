//
// Created by holynova on 2025/1/15.
//

#include "graph/nodes/SoftmaxLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    SoftmaxLayerNode::SoftmaxLayerNode(float beta) : _beta(beta)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    float SoftmaxLayerNode::beta() const
    {
        return _beta;
    }

    bool SoftmaxLayerNode::forward_descriptors()
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

    BITensorDescriptor SoftmaxLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor out_desc = src->desc();
        out_desc.quant_info         = get_softmax_output_quantization_info(out_desc.data_type, false);

        return out_desc;
    }

    BINodeType SoftmaxLayerNode::type() const
    {
        return BINodeType::SoftmaxLayer;
    }

    void SoftmaxLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
