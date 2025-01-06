//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_PRELULAYERNODE_H
#define BATMANINFER_GRAPH_PRELULAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** PRelu Layer node */
    class PReluLayerNode final : public BIINode
    {
    public:
        /** Default Constructor */
        PReluLayerNode();

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_PRELULAYERNODE_H
