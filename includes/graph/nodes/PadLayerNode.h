//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_PADLAYERNODE_H
#define BATMANINFER_GRAPH_PADLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Pad Layer node */
    class PadLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] padding   The padding for each spatial dimension of the input tensor. The pair padding[i]
         *                      specifies the front and the end padding in the i-th dimension.
         * @param[in] pad_value Padding value to be used. Defaults to 0
         */
        PadLayerNode(const PaddingList &padding, BIPixelValue pad_value = BIPixelValue());
        /** Padding list accessor
         *
         * @return Padding list
         */
        const PaddingList &padding() const;
        /** Padding value accessor
         *
         * @return Padding value
         */
        BIPixelValue pad_value() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool             forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void             accept(BIINodeVisitor &v) override;

    public:
        static constexpr BINodeType node_type = BINodeType::PadLayer;

    private:
        PaddingList   _padding;
        BIPixelValue  _pad_value;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_PADLAYERNODE_H
