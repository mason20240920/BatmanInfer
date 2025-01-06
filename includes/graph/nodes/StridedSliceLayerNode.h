//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_STRIDEDSLICELAYERNODE_H
#define BATMANINFER_GRAPH_STRIDEDSLICELAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Slice Layer node */
    class StridedSliceLayerNode final : public BIINode
    {
    public:
        /** Default Constructor
         *
         * @param[in] starts             The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] ends               The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] strides            The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] strided_slice_info Contains masks for the starts, ends and strides
         */
        StridedSliceLayerNode(const BICoordinates    &starts,
                              const BICoordinates    &ends,
                              const BiStrides        &strides,
                              BIStridedSliceLayerInfo strided_slice_info);
        /** Computes slice layer output descriptor
         *
         * @param[in] input_descriptor Descriptor of the input tensor
         * @param[in] starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] info             Contains masks for the starts, ends and strides
         *
         * @return  Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                            const BICoordinates      &starts,
                                                            const BICoordinates      &ends,
                                                            const BiStrides          &strides,
                                                            BIStridedSliceLayerInfo   info);
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
        /** Strides vector accessor
         *
         * @return End coordinates of the dimensions
         */
        BiStrides strides() const;

        BIStridedSliceLayerInfo strided_slice_info() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        BICoordinates           _starts;
        BICoordinates           _ends;
        BiStrides               _strides;
        BIStridedSliceLayerInfo _info;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_STRIDEDSLICELAYERNODE_H
