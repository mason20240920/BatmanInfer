//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/ROIAlignLayerNode.h"

#include "data/core/bi_helpers.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    ROIAlignLayerNode::ROIAlignLayerNode(BIROIPoolingLayerInfo &pool_info) : _pool_info(pool_info)
    {
        _input_edges.resize(2, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    const BIROIPoolingLayerInfo &ROIAlignLayerNode::pooling_info() const
    {
        return _pool_info;
    }

    bool ROIAlignLayerNode::forward_descriptors()
    {
        if ((input_id(0) != NullTensorID) && (input_id(1) != NullTensorID) && (output_id(0) != NullTensorID))
        {
            BITensor *dst = output(0);
            BI_COMPUTE_ERROR_ON(dst == nullptr);
            dst->desc() = configure_output(0);
            return true;
        }
        return false;
    }

    BITensorDescriptor ROIAlignLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src  = input(0);
        const BITensor *rois = input(1);
        BI_COMPUTE_ERROR_ON(src == nullptr);
        BI_COMPUTE_ERROR_ON(rois == nullptr);

        BITensorDescriptor output_desc = src->desc();

        // const size_t idx_n = get_data_layout_dimension_index(output_desc.layout, BIDataLayoutDimension::BATCHES);
        // const size_t idx_c = get_data_layout_dimension_index(output_desc.layout, BIDataLayoutDimension::CHANNEL);
        // const size_t idx_h = get_data_layout_dimension_index(output_desc.layout, BIDataLayoutDimension::HEIGHT);
        // const size_t idx_w = get_data_layout_dimension_index(output_desc.layout, BIDataLayoutDimension::WIDTH);
        const size_t idx_n = get_data_layout_dimension_index(BIDataLayoutDimension::BATCHES);
        const size_t idx_c = get_data_layout_dimension_index(BIDataLayoutDimension::CHANNEL);
        const size_t idx_h = get_data_layout_dimension_index(BIDataLayoutDimension::HEIGHT);
        const size_t idx_w = get_data_layout_dimension_index(BIDataLayoutDimension::WIDTH);

        output_desc.shape.set(idx_n, rois->desc().shape[1]);
        output_desc.shape.set(idx_c, src->desc().shape[idx_c]);
        output_desc.shape.set(idx_h, _pool_info.pooled_height());
        output_desc.shape.set(idx_w, _pool_info.pooled_width());

        return output_desc;
    }

    BINodeType ROIAlignLayerNode::type() const
    {
        return BINodeType::ROIAlignLayer;
    }

    void ROIAlignLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
