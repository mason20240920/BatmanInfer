//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_ACTIVATIONLAYERNODE_H
#define BATMANINFER_GRAPH_ACTIVATIONLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Activation Layer node */
    class ActivationLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] info           Activation Layer information
         * @param[in] out_quant_info (Optional) Output quantization info
         */
        ActivationLayerNode(BIActivationLayerInfo info, BIQuantizationInfo out_quant_info = BIQuantizationInfo());
        /** Activation metadata accessor
         *
         * @return The activation info of the layer
         */
        BIActivationLayerInfo activation_info() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    public:
        static constexpr BINodeType node_type = BINodeType::ActivationLayer;

    private:
        BIActivationLayerInfo _info;
        BIQuantizationInfo    _out_quant_info;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_ACTIVATIONLAYERNODE_H
