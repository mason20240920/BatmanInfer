//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/ResizeLayerNode.h"

#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    ResizeLayerNode::ResizeLayerNode(BIInterpolationPolicy policy, float scale_width, float scale_height)
        : _policy(policy), _scale_width(scale_width), _scale_height(scale_height)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BIInterpolationPolicy ResizeLayerNode::policy() const
    {
        return _policy;
    }

    std::pair<float, float> ResizeLayerNode::scaling_factor() const
    {
        return std::make_pair(_scale_width, _scale_height);
    }

    bool ResizeLayerNode::forward_descriptors()
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

    BITensorDescriptor ResizeLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        const BIDataLayout data_layout = src->desc().layout;
        BITensorDescriptor output_desc = src->desc();
        size_t           width_idx   = get_dimension_idx(data_layout, BIDataLayoutDimension::WIDTH);
        size_t           height_idx  = get_dimension_idx(data_layout, BIDataLayoutDimension::HEIGHT);
        output_desc.shape.set(width_idx, static_cast<int>(output_desc.shape[width_idx] * _scale_width));
        output_desc.shape.set(height_idx, static_cast<int>(output_desc.shape[height_idx] * _scale_height));

        return output_desc;
    }

    BINodeType ResizeLayerNode::type() const
    {
        return BINodeType::ResizeLayer;
    }

    void ResizeLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
