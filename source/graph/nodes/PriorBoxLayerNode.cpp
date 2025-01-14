//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/PriorBoxLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    PriorBoxLayerNode::PriorBoxLayerNode(BIPriorBoxLayerInfo prior_info) : _info(std::move(prior_info))
    {
        _input_edges.resize(2, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BIPriorBoxLayerInfo PriorBoxLayerNode::priorbox_info() const
    {
        return _info;
    }

    BITensorDescriptor PriorBoxLayerNode::compute_output_descriptor(const BITensorDescriptor  &input_descriptor,
                                                                    const BIPriorBoxLayerInfo &info)
    {
        const unsigned int layer_width  = get_dimension_size(input_descriptor, BIDataLayoutDimension::WIDTH);
        const unsigned int layer_height = get_dimension_size(input_descriptor, BIDataLayoutDimension::HEIGHT);
        const unsigned int num_priors   = info.aspect_ratios().size() * info.min_sizes().size() + info.max_sizes().size();

        BITensorDescriptor output_descriptor = input_descriptor;
        output_descriptor.shape.set(0, layer_width * layer_height * num_priors * 4);
        output_descriptor.shape.set(1, 2);
        output_descriptor.shape.set(2, 1);

        return output_descriptor;
    }

    bool PriorBoxLayerNode::forward_descriptors()
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

    BITensorDescriptor PriorBoxLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *input0 = input(0);
        BI_COMPUTE_ERROR_ON(input0 == nullptr);

        return compute_output_descriptor(input0->desc(), _info);
    }

    BINodeType PriorBoxLayerNode::type() const
    {
        return BINodeType::PriorBoxLayer;
    }

    void PriorBoxLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
