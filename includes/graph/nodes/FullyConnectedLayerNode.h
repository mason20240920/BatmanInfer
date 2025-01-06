//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_FULLYCONNECTEDLAYERNODE_H
#define BATMANINFER_GRAPH_FULLYCONNECTEDLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Fully Connected Layer node */
    class FullyConnectedLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] num_outputs    Number of neurons in the layer
         * @param[in] out_quant_info (Optional) Output quantization info
         * @param[in] fc_info        (Optional) Additional information about the fully connected layer
         * @param[in] fast_math_hint (Optional) Fast math hint
         */
        FullyConnectedLayerNode(unsigned int              num_outputs,
                                BIQuantizationInfo        out_quant_info = BIQuantizationInfo(),
                                BIFullyConnectedLayerInfo fc_info        = BIFullyConnectedLayerInfo(),
                                BIFastMathHint            fast_math_hint = BIFastMathHint::Disabled);
        /** Sets the fast math fast hint
         *
         * @param[in] hint Hint to use for fullyconnected layer
         */
        void set_fast_math_hint(BIFastMathHint hint);
        /** Fast math hint accessor
         *
         * @return Fast math hint to be used by the node
         */
        BIFastMathHint fast_math_hint() const;
        /** Sets fused activation
         *
         * @param[in] fused_activation Fused activation to set
         */
        void set_fused_activation(BIActivationLayerInfo fused_activation);
        /** Computes weights descriptor
         *
         * @warning Works for inputs with 1D batch space
         *
         * @param[in] input_descriptor   Input descriptor
         * @param[in] num_outputs        Number of output neurons
         * @param[in] fc_info            (Optional) Additional information about the fully connected layer
         * @param[in] weights_quant_info (Optional) Weights quantization info
         *
         * @return Weights descriptor
         */
        static BITensorDescriptor compute_weights_descriptor(const BITensorDescriptor &input_descriptor,
                                                             unsigned int              num_outputs,
                                                             BIFullyConnectedLayerInfo fc_info = BIFullyConnectedLayerInfo(),
                                                             const BIQuantizationInfo &weights_quant_info = BIQuantizationInfo());
        /** Computes fully connected layer output descriptor
         *
         * @warning Works for inputs with 1D batch space
         *
         * @param[in] input_descriptor Input descriptor
         * @param[in] num_outputs      Number of output neurons
         * @param[in] out_quant_info   (Optional) Weights quantization info
         *
         * @return Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                            unsigned int              num_outputs,
                                                            const BIQuantizationInfo &out_quant_info = BIQuantizationInfo());
        /** Fully connected layer addition information
         *
         * @return Additional information about the fully connected layer
         */
        BIFullyConnectedLayerInfo info() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

        static constexpr BINodeType node_type = BINodeType::FullyConnectedLayer;

    private:
        unsigned int              _num_outputs;
        BIQuantizationInfo        _out_quant_info;
        BIFullyConnectedLayerInfo _info;
        BIFastMathHint            _fast_math_hint;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_FULLYCONNECTEDLAYERNODE_H
