//
// Created by holynova on 2025/1/13.
//

#include "graph/nodes/ConcatenateLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "data/core/utils/misc/bi_shape_calculator.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    ConcatenateLayerNode::ConcatenateLayerNode(unsigned int                       total_nodes,
                                               descriptors::ConcatLayerDescriptor concat_descriptor)
    : _total_nodes(total_nodes), _concat_descriptor(std::move(concat_descriptor)), _is_enabled(true)
    {
        _input_edges.resize(_total_nodes, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    void ConcatenateLayerNode::set_enabled(bool is_enabled)
    {
        _is_enabled = is_enabled;
    }

    bool ConcatenateLayerNode::is_enabled() const
    {
        return _is_enabled;
    }

    BIDataLayoutDimension ConcatenateLayerNode::concatenation_axis() const
    {
        return _concat_descriptor.axis;
    }

    BIQuantizationInfo ConcatenateLayerNode::output_quantization_info() const
    {
        return _concat_descriptor.output_qinfo;
    }

    BITensorDescriptor ConcatenateLayerNode::compute_output_descriptor(
        const std::vector<BITensorDescriptor> &input_descriptors,
        BIDataLayoutDimension                  axis)
    {
        BI_COMPUTE_ERROR_ON(input_descriptors.size() == 0);

        BITensorDescriptor output_descriptor = input_descriptors[0];
        const int          axis_idx          = get_dimension_idx(output_descriptor.layout, axis);
        BI_COMPUTE_ERROR_ON_MSG(axis_idx > 2, "Unsupported concatenation axis!");

        // Extract shapes
        std::vector<const BITensorShape *> shapes;
        shapes.reserve(input_descriptors.size());
        for (auto &input_descriptor : input_descriptors)
        {
            shapes.emplace_back(&input_descriptor.shape);
        }

        output_descriptor.shape = BatmanInfer::misc::shape_calculator::calculate_concatenate_shape(shapes, axis_idx);

        return output_descriptor;
    }

    bool ConcatenateLayerNode::forward_descriptors()
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

    BITensorDescriptor ConcatenateLayerNode::configure_output(size_t idx) const
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
            output_info = compute_output_descriptor(inputs_descriptors, _concat_descriptor.axis);
            if (!_concat_descriptor.output_qinfo.empty())
            {
                output_info.quant_info = _concat_descriptor.output_qinfo;
            }
        }

        return output_info;
    }

    BINodeType ConcatenateLayerNode::type() const
    {
        return BINodeType::ConcatenateLayer;
    }

    void ConcatenateLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
