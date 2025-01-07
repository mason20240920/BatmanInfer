//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_SLICELAYERNODE_H
#define BATMANINFER_GRAPH_SLICELAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Slice Layer node */
    class SliceLayerNode final : public BIINode
    {
    public:
        /** Default Constructor
         *
         * @param[in] starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         */
        SliceLayerNode(const BICoordinates &starts, const BICoordinates &ends);
        /** Computes slice layer output descriptor
         *
         * @param[in] input_descriptor Descriptor of the input tensor
         * @param[in] starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         *
         * @return  Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                            const BICoordinates      &starts,
                                                            const BICoordinates      &ends);
        /** Start coordinates accessor
         *
         * @return Start coordinates of the dimensions
         */
        BICoordinates starts() const;
        /** End coordinates accessor
         *
         * @return End coordinates of the dimensions
         */
        BICoordinates ends() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BICoordinates _starts;
        BICoordinates _ends;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_SLICELAYERNODE_H
