//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/SliceLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "data/core/utils/misc/bi_shape_calculator.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    SliceLayerNode::SliceLayerNode(const BICoordinates &starts, const BICoordinates &ends) : _starts(starts), _ends(ends)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BICoordinates SliceLayerNode::starts() const
    {
        return _starts;
    }

    BICoordinates SliceLayerNode::ends() const
    {
        return _ends;
    }

    BITensorDescriptor SliceLayerNode::compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                                 const BICoordinates      &starts,
                                                                 const BICoordinates      &ends)
    {
        // using namespace BatmanInfer::helpers::tensor_transform;

        BITensorDescriptor output_desc = input_descriptor;
        output_desc.shape = BatmanInfer::misc::shape_calculator::compute_slice_shape(input_descriptor.shape, starts, ends);

        return output_desc;
    }

    bool SliceLayerNode::forward_descriptors()
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

    BITensorDescriptor SliceLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        return compute_output_descriptor(src->desc(), _starts, _ends);
    }

    BINodeType SliceLayerNode::type() const
    {
        return BINodeType::SliceLayer;
    }

    void SliceLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
