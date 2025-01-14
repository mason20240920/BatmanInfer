//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/ReductionLayerNode.h"

#include "data/core/utils/misc/bi_shape_calculator.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    ReductionLayerNode::ReductionLayerNode(BIReductionOperation op, unsigned int axis, bool keep_dims)
        : _op(op), _axis(axis), _keep_dims(keep_dims)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BIReductionOperation ReductionLayerNode::op() const
    {
        return _op;
    }

    unsigned int ReductionLayerNode::axis() const
    {
        return _axis;
    }

    bool ReductionLayerNode::keep_dims() const
    {
        return _keep_dims;
    }

    bool ReductionLayerNode::forward_descriptors()
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

    BITensorDescriptor ReductionLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor output_info = src->desc();
        BITensorShape      output_shape =
            BatmanInfer::misc::shape_calculator::compute_reduced_shape(output_info.shape, _axis, _keep_dims);
        output_info.set_shape(output_shape);

        return output_info;
    }

    BINodeType ReductionLayerNode::type() const
    {
        return BINodeType::ReductionOperationLayer;
    }

    void ReductionLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
