//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/DetectionPostProcessLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    DetectionPostProcessLayerNode::DetectionPostProcessLayerNode(BIDetectionPostProcessLayerInfo detection_info)
    : _info(detection_info)
    {
        _input_edges.resize(3, EmptyEdgeID);
        _outputs.resize(4, NullTensorID);
    }

    BIDetectionPostProcessLayerInfo DetectionPostProcessLayerNode::detection_post_process_info() const
    {
        return _info;
    }

    bool DetectionPostProcessLayerNode::forward_descriptors()
    {
        if ((input_id(0) != NullTensorID) && (input_id(1) != NullTensorID) && (input_id(2) != NullTensorID) &&
            (output_id(0) != NullTensorID) && (output_id(1) != NullTensorID) && (output_id(2) != NullTensorID) &&
            (output_id(3) != NullTensorID))
        {
            for (unsigned int i = 0; i < 4; ++i)
            {
                BITensor *dst = output(i);
                BI_COMPUTE_ERROR_ON(dst == nullptr);
                dst->desc() = configure_output(i);
            }
            return true;
        }
        return false;
    }

    BITensorDescriptor DetectionPostProcessLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        BITensorDescriptor   output_desc;
        const unsigned int num_detected_box = _info.max_detections() * _info.max_classes_per_detection();

        switch (idx)
        {
            case 0:
                // Configure boxes output
                output_desc.shape = BITensorShape(kNumCoordBox, num_detected_box, kBatchSize);
                break;
            case 1:
            case 2:
                // Configure classes or scores output
                output_desc.shape = BITensorShape(num_detected_box, kBatchSize);
                break;
            case 3:
                // Configure num_detection
                output_desc.shape = BITensorShape(1);
                break;
            default:
                BI_COMPUTE_ERROR("Unsupported output index");
        }
        output_desc.data_type = BIDataType::F32;

        return output_desc;
    }

    BINodeType DetectionPostProcessLayerNode::type() const
    {
        return BINodeType::DetectionPostProcessLayer;
    }

    void DetectionPostProcessLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
