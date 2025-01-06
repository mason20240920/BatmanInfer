//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_NORMALIZEPLANARYUVLAYERNODE_H
#define BATMANINFER_GRAPH_NORMALIZEPLANARYUVLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Batch Normalization Layer node */
    class NormalizePlanarYUVLayerNode final : public BIINode
    {
    public:
        /** Constructor */
        NormalizePlanarYUVLayerNode();

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_NORMALIZEPLANARYUVLAYERNODE_H
