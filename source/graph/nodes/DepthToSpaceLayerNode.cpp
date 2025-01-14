//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/DepthToSpaceLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "data/core/utils/misc/bi_shape_calculator.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    DepthToSpaceLayerNode::DepthToSpaceLayerNode(int block_shape) : _block_shape(block_shape)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    int DepthToSpaceLayerNode::block_shape() const
    {
        return _block_shape;
    }

    BITensorDescriptor DepthToSpaceLayerNode::compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                                        int                       block_shape)
    {
        // using namespace BatmanInfer::helpers::tensor_transform;

        BITensorShape input_shape = input_descriptor.shape;
        BIDataLayout  data_layout = input_descriptor.layout;

        // Set descriptor shape
        BITensorDescriptor output_descriptor = input_descriptor;
        output_descriptor.shape =
            misc::shape_calculator::compute_depth_to_space_shape(input_shape, data_layout, block_shape);

        return output_descriptor;
    }

    bool DepthToSpaceLayerNode::forward_descriptors()
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

    BITensorDescriptor DepthToSpaceLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor output_desc = src->desc();

        return compute_output_descriptor(output_desc, _block_shape);
    }

    BINodeType DepthToSpaceLayerNode::type() const
    {
        return BINodeType::DepthToSpaceLayer;
    }

    void DepthToSpaceLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
