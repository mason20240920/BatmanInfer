//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_DUMMYNODE_H
#define BATMANINFER_GRAPH_DUMMYNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Dummy Layer node
     *
     * Dummy layer transforms a given input to a specified output with a given shape.
     *
     * @note Used only for debugging/performance reasons.
     * @note It does not perform any computation at all.
     * @note Can be used to simulate graphs that they have nodes that are not yet supported.
     */
    class DummyNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] shape Reshaped tensor shape
         */
        DummyNode(BITensorShape shape);

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

#endif //BATMANINFER_GRAPH_DUMMYNODE_H
