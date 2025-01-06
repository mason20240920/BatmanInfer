//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_CONSTNODE_H
#define BATMANINFER_GRAPH_CONSTNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Const node */
    class ConstNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] desc Tensor descriptor
         */
        ConstNode(BITensorDescriptor desc);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BITensorDescriptor _desc;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_CONSTNODE_H
