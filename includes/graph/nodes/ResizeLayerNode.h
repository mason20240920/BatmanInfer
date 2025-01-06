//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_RESIZELAYERNODE_H
#define BATMANINFER_GRAPH_RESIZELAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Resize Layer node */
    class ResizeLayerNode final : public BIINode
    {
    public:
        /** Default Constructor */
        ResizeLayerNode(BIInterpolationPolicy policy, float scale_width, float scale_height);
        /** Interpolation policy accessor
         *
         * @return Interpolation policy
         */
        BIInterpolationPolicy policy() const;
        /** Scaling factor accessors
         *
         * @return Scaling factors (width, height)
         */
        std::pair<float, float> scaling_factor() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BIInterpolationPolicy _policy;
        float                 _scale_width;
        float                 _scale_height;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_RESIZELAYERNODE_H
