//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_PERMUTELAYERNODE_H
#define BATMANINFER_GRAPH_PERMUTELAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Permute Layer node */
    class PermuteLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] perm   Permutation vector
         * @param[in] layout (Optional) Data layout to assign to permuted tensor.
         *                   If UNKNOWN then the input's layout will be used.
         */
        PermuteLayerNode(PermutationVector perm, BIDataLayout layout = BIDataLayout::UNKNOWN);
        /** Permutation vector accessor
         *
         * @return Permutation vector
         */
        const PermutationVector &permutation_vector() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        PermutationVector _perm;
        BIDataLayout      _layout;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_PERMUTELAYERNODE_H
