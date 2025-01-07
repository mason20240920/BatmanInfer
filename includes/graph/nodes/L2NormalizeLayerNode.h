//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_L2NORMALIZELAYERNODE_H
#define BATMANINFER_GRAPH_L2NORMALIZELAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** L2Normalize Layer node */
    class L2NormalizeLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         */
        L2NormalizeLayerNode();

        /** Constructor
         *
         * @param[in] axis Axis to perform normalization on
         */
        L2NormalizeLayerNode(int axis);

        /** Constructor
         *
         * @param[in] axis    Axis to perform normalization on
         * @param[in] epsilon Lower bound value for the normalization
         */
        L2NormalizeLayerNode(int axis, float epsilon);

        /** axis accessors
         *
         * @return Axis to perform normalization on
         */
        int axis() const;

        /** epsilon accessors
         *
         * @return Lower bound value for the normalization
         */
        float epsilon() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        int   _axis;
        float _epsilon;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_L2NORMALIZELAYERNODE_H
