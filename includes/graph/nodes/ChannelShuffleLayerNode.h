//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_CHANNELSHUFFLELAYERNODE_H
#define BATMANINFER_GRAPH_CHANNELSHUFFLELAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Channel Shuffle Layer node */
    class ChannelShuffleLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] num_groups Number of groups
         */
        ChannelShuffleLayerNode(unsigned int num_groups);
        /** Number of groups accessor
         *
         * @return Number of groups
         */
        unsigned int num_groups() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        unsigned int _num_groups;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_CHANNELSHUFFLELAYERNODE_H
