//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/QuantizationLayerNode.h"

#include "data/core/utils/data_type_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    QuantizationLayerNode::QuantizationLayerNode(BIQuantizationInfo out_quant_info)
        : QuantizationLayerNode(out_quant_info, BIDataType::QASYMM8)
    {
    }

    QuantizationLayerNode::QuantizationLayerNode(BIQuantizationInfo out_quant_info, BIDataType out_data_type)
        : _out_quant_info(std::move(out_quant_info)), _out_data_type(out_data_type)
    {
        BI_COMPUTE_ERROR_ON(!is_data_type_quantized(out_data_type));

        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    bool QuantizationLayerNode::forward_descriptors()
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

    BITensorDescriptor QuantizationLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        BITensorDescriptor output_info = src->desc();
        output_info.data_type        = _out_data_type;
        output_info.quant_info       = _out_quant_info;

        return output_info;
    }

    BINodeType QuantizationLayerNode::type() const
    {
        return QuantizationLayerNode::node_type;
    }

    void QuantizationLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
