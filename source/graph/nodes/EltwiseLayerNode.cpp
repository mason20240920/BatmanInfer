//
// Created by holynova on 2025/1/14.
//

#include "graph/nodes/EltwiseLayerNode.h"

#include "data/bi_tensor_shape.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_inodeVisitor.h"

namespace BatmanInfer {

namespace graph {

    EltwiseLayerNode::EltwiseLayerNode(const descriptors::EltwiseLayerDescriptor &descriptor) : descriptor(descriptor)
    {
        _input_edges.resize(2, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    BIEltwiseOperation EltwiseLayerNode::eltwise_operation() const
    {
        return descriptor.op;
    }

    BIConvertPolicy EltwiseLayerNode::convert_policy() const
    {
        return descriptor.c_policy;
    }

    BIRoundingPolicy EltwiseLayerNode::rounding_policy() const
    {
        return descriptor.r_policy;
    }

    BIActivationLayerInfo EltwiseLayerNode::fused_activation() const
    {
        return descriptor.fused_activation;
    }

    BIQuantizationInfo EltwiseLayerNode::output_quant_info() const
    {
        return descriptor.out_quant_info;
    }

    void EltwiseLayerNode::set_fused_activation(BIActivationLayerInfo fused_activation)
    {
        descriptor.fused_activation = fused_activation;
    }

    bool EltwiseLayerNode::forward_descriptors()
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

    BITensorDescriptor EltwiseLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);

        const BITensor *src1 = input(0);
        BI_COMPUTE_ERROR_ON(src1 == nullptr);

        const BITensor *src2 = input(1);
        BI_COMPUTE_ERROR_ON(src2 == nullptr);

        auto output_info = src1->desc();

        BITensorShape out_shape = BITensorShape::broadcast_shape(src1->desc().shape, src2->desc().shape);
        BI_COMPUTE_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

        output_info.set_shape(out_shape);

        if (!descriptor.out_quant_info.empty())
        {
            output_info.set_quantization_info(descriptor.out_quant_info);
        }

        return output_info;
    }

    BINodeType EltwiseLayerNode::type() const
    {
        return BINodeType::EltwiseLayer;
    }

    void EltwiseLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

    UnaryEltwiseLayerNode::UnaryEltwiseLayerNode(const descriptors::UnaryEltwiseLayerDescriptor &descriptor)
        : descriptor(descriptor)
    {
        _input_edges.resize(1, EmptyEdgeID);
        _outputs.resize(1, NullTensorID);
    }

    descriptors::UnaryEltwiseLayerDescriptor UnaryEltwiseLayerNode::eltwise_descriptor() const
    {
        return descriptor;
    }

    void UnaryEltwiseLayerNode::set_fused_activation(BIActivationLayerInfo fused_activation)
    {
        descriptor.fused_activation = fused_activation;
    }

    bool UnaryEltwiseLayerNode::forward_descriptors()
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

    BITensorDescriptor UnaryEltwiseLayerNode::configure_output(size_t idx) const
    {
        BI_COMPUTE_UNUSED(idx);

        const BITensor *src = input(0);
        BI_COMPUTE_ERROR_ON(src == nullptr);

        auto output_info = src->desc();

        if (!descriptor.out_quant_info.empty())
        {
            output_info.set_quantization_info(descriptor.out_quant_info);
        }

        return output_info;
    }

    BINodeType UnaryEltwiseLayerNode::type() const
    {
        return BINodeType::UnaryEltwiseLayer;
    }

    void UnaryEltwiseLayerNode::accept(BIINodeVisitor &v)
    {
        v.visit(*this);
    }

} // namespace graph

} // namespace BatmanInfer
