//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_RESHAPELAYERNODE_H
#define BATMANINFER_GRAPH_RESHAPELAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Reshape Layer node */
    class ReshapeLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] shape Reshaped tensor shape
         */
        ReshapeLayerNode(BITensorShape shape);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BITensorShape _shape;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_RESHAPELAYERNODE_H
