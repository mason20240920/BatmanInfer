//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_NORMALIZATIONLAYERNODE_H
#define BATMANINFER_GRAPH_NORMALIZATIONLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Normalization Layer node */
    class NormalizationLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] norm_info Normalization Layer information
         */
        NormalizationLayerNode(BINormalizationLayerInfo norm_info);
        /** Normalization info accessor
         *
         * @return Normalization layer info
         */
        BINormalizationLayerInfo normalization_info() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BINormalizationLayerInfo _info;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_NORMALIZATIONLAYERNODE_H
