//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_ROIALIGNLAYERNODE_H
#define BATMANINFER_GRAPH_ROIALIGNLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** ROI Align node */
    class ROIAlignLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] pool_info Contains pooling operation information described in @ref ROIPoolingLayerInfo.
         */
        ROIAlignLayerNode(BIROIPoolingLayerInfo &pool_info);
        /** Prevent instances of this class from being copy constructed */
        ROIAlignLayerNode(const ROIAlignLayerNode &) = delete;
        /** Prevent instances of this class from being copied */
        ROIAlignLayerNode &operator=(const ROIAlignLayerNode &) = delete;

        /** ROIPoolingLayerInfo accessor
         *
         * @return ROIPoolingLayerInfo
         */
        const BIROIPoolingLayerInfo &pooling_info() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BIROIPoolingLayerInfo _pool_info;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_ROIALIGNLAYERNODE_H
