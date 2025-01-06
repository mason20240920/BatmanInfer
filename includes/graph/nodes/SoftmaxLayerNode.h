//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_SOFTMAXLAYERNODE_H
#define BATMANINFER_GRAPH_SOFTMAXLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Softmax Layer node */
    class SoftmaxLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] beta (Optional) Beta parameter. Defaults to 1
         */
        SoftmaxLayerNode(float beta = 1.f);
        /** Beta parameter accessor
         *
         * @return Beta parameter
         */
        float beta() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    public:
        static constexpr BINodeType node_type = BINodeType::SoftmaxLayer;

    private:
        float _beta;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_SOFTMAXLAYERNODE_H
