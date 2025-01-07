//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_ELTWISELAYERNODE_H
#define BATMANINFER_GRAPH_ELTWISELAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Eltwise Layer node */
    class EltwiseLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] descriptor Containing information for the node described in @ref descriptors::EltwiseLayerDescriptor
         */
        EltwiseLayerNode(const descriptors::EltwiseLayerDescriptor &descriptor);
        /** Eltwise operation accessor
         *
         * @return Eltwise operation that is to be performed by the node
         */
        BIEltwiseOperation eltwise_operation() const;

        /** Convert policy accessor
         *
         * @return Convert policy that is used in the node
         */
        BIConvertPolicy convert_policy() const;

        /** Rounding policy accessor
         *
         * @return Convert policy that is used in the node
         */
        BIRoundingPolicy rounding_policy() const;

        /** Returns fused activation
         *
         * @return Fused activation
         */
        BIActivationLayerInfo fused_activation() const;

        /** Returns output quantization info
         *
         * @return Output quantization info
         */
        BIQuantizationInfo output_quant_info() const;

        /** Sets fused activation
         *
         * @param[in] fused_activation Fused activation to set
         */
        void set_fused_activation(BIActivationLayerInfo fused_activation);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

        static constexpr BINodeType node_type = BINodeType::EltwiseLayer;

    private:
        descriptors::EltwiseLayerDescriptor descriptor;
    };

/** Unary Eltwise Layer node */
    class UnaryEltwiseLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] descriptor Containing information for the node described in @ref descriptors::EltwiseLayerDescriptor
         */
        UnaryEltwiseLayerNode(const descriptors::UnaryEltwiseLayerDescriptor &descriptor);
        /** Unary eltwise layer descriptor
         *
         * @return Unary eltwise layer descriptor which containing information
         */
        descriptors::UnaryEltwiseLayerDescriptor eltwise_descriptor() const;

        /** Sets fused activation
         *
         * @param[in] fused_activation Fused activation to set
         */
        void set_fused_activation(BIActivationLayerInfo fused_activation);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

        static constexpr BINodeType node_type = BINodeType::UnaryEltwiseLayer;

    private:
        descriptors::UnaryEltwiseLayerDescriptor descriptor;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_ELTWISELAYERNODE_H
