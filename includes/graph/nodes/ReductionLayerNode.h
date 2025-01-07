//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_REDUCTIONLAYERNODE_H
#define BATMANINFER_GRAPH_REDUCTIONLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Reduction Operation node */
    class ReductionLayerNode final : public BIINode
    {
    public:
        /** Default Constructor */
        ReductionLayerNode(BIReductionOperation op, unsigned int axis, bool keep_dims = true);
        /** op accessor
         *
         * @return op
         */
        BIReductionOperation op() const;
        /** axis accessor
         *
         * @return axis
         */
        unsigned int axis() const;
        /** keep_dims accessor
         *
         * @return keep_dims
         */
        bool keep_dims() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BIReductionOperation _op;
        unsigned int         _axis;
        bool                 _keep_dims;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_REDUCTIONLAYERNODE_H
