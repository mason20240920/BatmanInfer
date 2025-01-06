//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_BATCHNORMALIZATIONLAYERNODE_H
#define BATMANINFER_GRAPH_BATCHNORMALIZATIONLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Batch Normalization Layer node */
    class BatchNormalizationLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] epsilon          (Optional) Epsilon parameter. Defaults to 1.f
         * @param[in] fused_activation (Optional) Fused activation layer. Disabled if not specified
         */
        BatchNormalizationLayerNode(float epsilon = 1.f, BIActivationLayerInfo fused_activation = BIActivationLayerInfo());
        /** Epsilon parameter accessor
         *
         * @return Epsilon parameter
         */
        float epsilon() const;
        /** Returns fused activation
         *
         * @return Fused activation
         */
        BIActivationLayerInfo fused_activation() const;
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

    public:
        static constexpr BINodeType node_type = BINodeType::BatchNormalizationLayer;

    private:
        float                 _epsilon;
        BIActivationLayerInfo _fused_activation;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BATCHNORMALIZATIONLAYERNODE_H
