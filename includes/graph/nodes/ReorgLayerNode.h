//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_REORGLAYERNODE_H
#define BATMANINFER_GRAPH_REORGLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Reorg Layer node */
    class ReorgLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] stride Stride value to use for reorganizing the values in the output tensor.
         */
        ReorgLayerNode(int stride);
        /** Stride value to use for reorganizing the values in the output tensor.
         *
         * @return Stride value to use for reorganizing the values in the output tensor.
         */
        int stride() const;
        /** Computes reorg output descriptor
         *
         * @param[in] input_descriptor Input descriptor
         * @param[in] stride           Stride value to use for reorganizing the values in the output tensor.
         *
         * @return Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor &input_descriptor, int stride);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        int _stride;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_REORGLAYERNODE_H
