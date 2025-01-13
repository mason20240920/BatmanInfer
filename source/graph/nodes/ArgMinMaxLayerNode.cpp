//
// Created by holynova on 2025/1/10.
//

#include "graph/nodes/ArgMinMaxLayerNode.h"

#include "data/core/utils/misc/bi_shape_calculator.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    ArgMinMaxLayerNode::ArgMinMaxLayerNode(BIReductionOperation op,
                                           unsigned int       axis,
                                           BIDataType           out_data_type,
                                           BIQuantizationInfo   out_quant_info)
    : _op(op), _axis(axis), _out_data_type(out_data_type), _out_quant_info(std::move(out_quant_info))
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BIReductionOperation ArgMinMaxLayerNode::reduction_operation() const
    {
        return _op;
    }

    unsigned int ArgMinMaxLayerNode::axis() const
    {
        return _axis;
    }

    BIDataType ArgMinMaxLayerNode::out_data_type() const
    {
        return _out_data_type;
    }

    bool ArgMinMaxLayerNode::forward_descriptors()
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

    BITensorDescriptor ArgMinMaxLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor output_info = src->desc();
        if (!_out_quant_info.empty())
        {
            output_info.quant_info = _out_quant_info;
        }

        if (_out_data_type != BIDataType::UNKNOWN)
        {
            output_info.data_type = _out_data_type;
        }

        BITensorShape output_shape =
            BatmanInfer::misc::shape_calculator::compute_reduced_shape(output_info.shape, _axis, false);
        output_info.set_shape(output_shape);

        return output_info;
    }

} // namespace graph

} // namespace BatmanInfer
