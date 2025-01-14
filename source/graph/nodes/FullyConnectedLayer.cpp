//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/FullyConnectedLayerNode.h"

#include "data/core/bi_utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    FullyConnectedLayerNode::FullyConnectedLayerNode(unsigned int              num_outputs,
                                                     BIQuantizationInfo        out_quant_info,
                                                     BIFullyConnectedLayerInfo fc_info,
                                                     BIFastMathHint            fast_math_hint)
    : _num_outputs(num_outputs),
      _out_quant_info(std::move(out_quant_info)),
      _info(fc_info),
      _fast_math_hint(fast_math_hint)
    {
        _input_edges.resize(3, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    void FullyConnectedLayerNode::set_fast_math_hint(BIFastMathHint hint)
    {
        _fast_math_hint = hint;
    }

    BIFastMathHint FullyConnectedLayerNode::fast_math_hint() const
    {
        return _fast_math_hint;
    }

    void FullyConnectedLayerNode::set_fused_activation(BIActivationLayerInfo fused_activation)
    {
        _info.activation_info = fused_activation;
    }

    BITensorDescriptor FullyConnectedLayerNode::compute_weights_descriptor(
        const BITensorDescriptor &input_descriptor,
        unsigned int              num_outputs,
        BIFullyConnectedLayerInfo fc_info,
        const BIQuantizationInfo &weights_quant_info)
    {
        unsigned int num_weights    = 1;
        unsigned int num_dimensions = input_descriptor.shape.num_dimensions();
        // Ignore the batch dimension if there is one:
        if (num_dimensions == 2 || num_dimensions == 4)
        {
            num_dimensions--;
        }
        for (unsigned int i = 0; i < num_dimensions; i++)
        {
            num_weights *= input_descriptor.shape[i];
        }

        BITensorDescriptor weights_descriptor = input_descriptor;
        weights_descriptor.shape              = BITensorShape(num_weights, num_outputs);

        // If weights are tranposed, use tranposed shape
        if (!fc_info.transpose_weights)
        {
            weights_descriptor.shape = BITensorShape(num_outputs, num_weights);
        }

        // Set quantization info if present
        if (!weights_quant_info.empty())
        {
            weights_descriptor.quant_info = weights_quant_info;
        }

        return weights_descriptor;
    }

    BITensorDescriptor FullyConnectedLayerNode::compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                                          unsigned int              num_outputs,
                                                                          const BIQuantizationInfo &out_quant_info)
    {
        // Note: Only 1D batch space is supported at the moment
        unsigned int batches = input_descriptor.shape[1];
        if (input_descriptor.shape.num_dimensions() > 2)
        {
            batches = input_descriptor.shape[3];
        }

        // Set descriptor shape
        BITensorDescriptor output_descriptor = input_descriptor;
        output_descriptor.shape              = BITensorShape(num_outputs, batches);

        // Set quantization info if present
        if (!out_quant_info.empty())
        {
            output_descriptor.quant_info = out_quant_info;
        }

        return output_descriptor;
    }

    BIFullyConnectedLayerInfo FullyConnectedLayerNode::info() const
    {
        return _info;
    }

    bool FullyConnectedLayerNode::forward_descriptors()
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

    BITensorDescriptor FullyConnectedLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);
        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        return compute_output_descriptor(src->desc(), _num_outputs, _out_quant_info);
    }

    BINodeType FullyConnectedLayerNode::type() const
    {
        return BINodeType::FullyConnectedLayer;
    }

    void FullyConnectedLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
