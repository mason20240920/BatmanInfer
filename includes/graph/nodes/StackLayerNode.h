//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_STACKLAYERNODE_H
#define BATMANINFER_GRAPH_STACKLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Stack Layer node */
    class StackLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] total_nodes Number of nodes that will get stacked
         * @param[in] axis        Axis alogn which to stack the input tensors
         */
        StackLayerNode(unsigned int total_nodes, int axis);
        /** Computes stack output descriptor
         *
         * @param[in] input_descriptors Input descriptors
         * @param[in] axis              Axis along which to stack the input tensors
         *
         * @return Expected output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const std::vector<BITensorDescriptor> &input_descriptors, int axis);
        /** Stack axis parameter accessor
         *
         * @return Stack axis
         */
        int axis() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        unsigned int _total_nodes;
        int          _axis;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_STACKLAYERNODE_H
