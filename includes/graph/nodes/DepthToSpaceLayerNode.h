//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_DEPTHTOSPACELAYERNODE_H
#define BATMANINFER_GRAPH_DEPTHTOSPACELAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** DepthToSpace Layer node */
    class DepthToSpaceLayerNode final : public BIINode
    {
    public:
        /** Default Constructor */
        DepthToSpaceLayerNode(int block_shape);
        /** Block shape policy accessor
         *
         * @return Block shape
         */
        int block_shape() const;
        /** Computes depth to space output descriptor
         *
         * @warning block_shape must be greater than or equal to 2
         *
         * @param[in] input_descriptor Input descriptor
         * @param[in] block_shape      Number of output neurons
         *
         * @return Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor &input_descriptor, int block_shape);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        int _block_shape;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_DEPTHTOSPACELAYERNODE_H
