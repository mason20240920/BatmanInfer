//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_POOLINGLAYERNODE_H
#define BATMANINFER_GRAPH_POOLINGLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Pooling Layer node */
    class PoolingLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] pool_info Pooling Layer information
         */
        PoolingLayerNode(BIPoolingLayerInfo pool_info);
        /** Pooling metadata accessor
         *
         * @return Pooling Layer info
         */
        BIPoolingLayerInfo pooling_info() const;
        /** Computes pooling output descriptor
         *
         * @param[in] input_descriptor Input descriptor
         * @param[in] info             Pooling operation attributes
         *
         * @return Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor &input_descriptor, BIPoolingLayerInfo info);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BIPoolingLayerInfo _info;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_POOLINGLAYERNODE_H
