//
// Created by holynova on 2025/1/13.
//

#include "graph/nodes/BatchNormalizationLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    BatchNormalizationLayerNode::BatchNormalizationLayerNode(float epsilon, BIActivationLayerInfo fused_activation)
    : _epsilon(epsilon), _fused_activation(fused_activation)
    {
        _input_edges.resize(5, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    float BatchNormalizationLayerNode::epsilon() const
    {
        return _epsilon;
    }

    BIActivationLayerInfo BatchNormalizationLayerNode::fused_activation() const
    {
        return _fused_activation;
    }

    void BatchNormalizationLayerNode::set_fused_activation(BIActivationLayerInfo fused_activation)
    {
        _fused_activation = fused_activation;
    }

    bool BatchNormalizationLayerNode::forward_descriptors()
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

    BITensorDescriptor BatchNormalizationLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        return src->desc();
    }

    BINodeType BatchNormalizationLayerNode::type() const
    {
        return BatchNormalizationLayerNode::node_type;
    }

    void BatchNormalizationLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
