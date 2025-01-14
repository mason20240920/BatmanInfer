//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/PoolingLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    PoolingLayerNode::PoolingLayerNode(BIPoolingLayerInfo pool_info) : _info(std::move(pool_info))
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BIPoolingLayerInfo PoolingLayerNode::pooling_info() const
    {
        return _info;
    }

    BITensorDescriptor PoolingLayerNode::compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                                   BIPoolingLayerInfo        info)
    {
        unsigned int pooled_width  = 0;
        unsigned int pooled_height = 0;

        const unsigned int input_width  = get_dimension_size(input_descriptor, BIDataLayoutDimension::WIDTH);
        const unsigned int input_height = get_dimension_size(input_descriptor, BIDataLayoutDimension::HEIGHT);
        const unsigned int pool_size_x  = info.is_global_pooling ? input_width : info.pool_size.width;
        const unsigned int pool_size_y  = info.is_global_pooling ? input_height : info.pool_size.height;

        std::tie(pooled_width, pooled_height) =
            scaled_dimensions(input_width, input_height, pool_size_x, pool_size_y, info.pad_stride_info);

        const BIDataLayout data_layout       = input_descriptor.layout;
        BITensorDescriptor output_descriptor = input_descriptor;
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::WIDTH), pooled_width);
        output_descriptor.shape.set(get_dimension_idx(data_layout, BIDataLayoutDimension::HEIGHT), pooled_height);

        return output_descriptor;
    }

    bool PoolingLayerNode::forward_descriptors()
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

    BITensorDescriptor PoolingLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        return compute_output_descriptor(src->desc(), _info);
    }

    BINodeType PoolingLayerNode::type() const
    {
        return BINodeType::PoolingLayer;
    }

    void PoolingLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
