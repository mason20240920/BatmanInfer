//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/DetectionOutputLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    DetectionOutputLayerNode::DetectionOutputLayerNode(BIDetectionOutputLayerInfo detection_info) : _info(detection_info)
    {
        _input_edges.resize(3, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BIDetectionOutputLayerInfo DetectionOutputLayerNode::detection_output_info() const
    {
        return _info;
    }

    BITensorDescriptor DetectionOutputLayerNode::compute_output_descriptor(
        const BITensorDescriptor         &input_descriptor,
        const BIDetectionOutputLayerInfo &info)
    {
        const unsigned int max_size =
            info.keep_top_k() * ((input_descriptor.shape.num_dimensions() > 1) ? input_descriptor.shape[1] : 1);

        BITensorDescriptor output_descriptor = input_descriptor;
        output_descriptor.shape.set(0, detection_size);
        output_descriptor.shape.set(1, max_size);

        return output_descriptor;
    }

    bool DetectionOutputLayerNode::forward_descriptors()
    {
        if ((input_id(0) != NullTensorID) && (input_id(1) != NullTensorID) && (input_id(2) != NullTensorID) &&
            (output_id(0) != NullTensorID))
        {
            BITensor *dst = output(0);
            BI_COMPUTE_ERROR_ON(dst == nullptr);
            dst->desc() = configure_output(0);
            return true;
        }
        return false;
    }

    BITensorDescriptor DetectionOutputLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *input0 = input(0);
        BI_COMPUTE_ERROR_ON(input0 == nullptr);

        return compute_output_descriptor(input0->desc(), _info);
    }

    BINodeType DetectionOutputLayerNode::type() const
    {
        return BINodeType::DetectionOutputLayer;
    }

    void DetectionOutputLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
