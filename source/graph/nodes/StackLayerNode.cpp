//
// Created by holynova on 2025/1/15.
//

#include "graph/nodes/StackLayerNode.h"

#include "data/core/bi_tensor_info.hpp"
#include "data/core/bi_utils.hpp"
#include "data/core/utils/misc/bi_shape_calculator.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    StackLayerNode::StackLayerNode(unsigned int total_nodes, int axis) : _total_nodes(total_nodes), _axis(axis)
    {
        _input_edges.resize(_total_nodes, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    int StackLayerNode::axis() const
    {
        return _axis;
    }

    BITensorDescriptor StackLayerNode::compute_output_descriptor(const std::vector<BITensorDescriptor> &input_descriptors,
                                                                 int                                    axis)
    {
        BI_COMPUTE_ERROR_ON(input_descriptors.size() == 0);

        BITensorDescriptor output_descriptor = input_descriptors[0];

        const BITensorInfo   input_info(input_descriptors[0].shape, 1, input_descriptors[0].data_type);
        const unsigned int num_tensors = input_descriptors.size();

        output_descriptor.shape = BatmanInfer::misc::shape_calculator::compute_stack_shape(input_info, axis, num_tensors);

        return output_descriptor;
    }

    bool StackLayerNode::forward_descriptors()
    {
        if (_outputs[0] != NullTensorID)
        {
            BITensor *dst = output(0);
            BI_COMPUTE_ERROR_ON(dst == nullptr);
            dst->desc() = configure_output(0);
            return true;
        }
        return false;
    }

    BITensorDescriptor StackLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        // Check if all input tensors are set
        bool are_all_inputs_set = std::all_of(std::begin(_input_edges), std::end(_input_edges),
                                              [](const EdgeID &eid) { return eid != EmptyEdgeID; });

        BITensorDescriptor output_info = {};

        if (are_all_inputs_set)
        {
            std::vector<BITensorDescriptor> inputs_descriptors;
            for (unsigned int i = 0; i < _input_edges.size(); ++i)
            {
                const BITensor *t = _graph->tensor(input_id(i));
                BI_COMPUTE_ERROR_ON(t == nullptr);
                inputs_descriptors.push_back(t->desc());
            }
            output_info = compute_output_descriptor(inputs_descriptors, _axis);
        }

        return output_info;
    }

    BINodeType StackLayerNode::type() const
    {
        return BINodeType::StackLayer;
    }

    void StackLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
