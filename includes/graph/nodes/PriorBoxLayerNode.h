//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_PRIORBOXLAYERNODE_H
#define BATMANINFER_GRAPH_PRIORBOXLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** PriorBox Layer node */
    class PriorBoxLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] prior_info PriorBox Layer information
         */
        PriorBoxLayerNode(BIPriorBoxLayerInfo prior_info);
        /** PriorBox metadata accessor
         *
         * @return PriorBox Layer info
         */
        BIPriorBoxLayerInfo priorbox_info() const;
        /** Computes priorbox output descriptor
         *
         * @param[in] input_descriptor Input descriptor
         * @param[in] info             PriorBox operation attributes
         *
         * @return Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor  &input_descriptor,
                                                            const BIPriorBoxLayerInfo &info);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BIPriorBoxLayerInfo _info;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_PRIORBOXLAYERNODE_H
