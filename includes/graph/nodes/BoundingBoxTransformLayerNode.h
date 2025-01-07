//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_BOUNDINGBOXTRANSFORMLAYERNODE_H
#define BATMANINFER_GRAPH_BOUNDINGBOXTRANSFORMLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Bounding Box Transform Layer node */
    class BoundingBoxTransformLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] info Contains BoundingBox operation information described in @ref BoundingBoxTransformInfo.
         */
        BoundingBoxTransformLayerNode(BIBoundingBoxTransformInfo &info);
        /** BoundingBoxTransformInfo accessor
         *
         * @return BoundingBoxTransformInfo
         */
        const BIBoundingBoxTransformInfo &info() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BIBoundingBoxTransformInfo _bbox_info;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BOUNDINGBOXTRANSFORMLAYERNODE_H
