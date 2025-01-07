//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_DECONVOLUTIONLAYERNODE_H
#define BATMANINFER_GRAPH_DECONVOLUTIONLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Deconvolution Layer node */
    class DeconvolutionLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] descriptor Contains information used by this layer described in @ref descriptors::DeconvolutionLayerDescriptor
         */
        DeconvolutionLayerNode(const descriptors::DeconvolutionLayerDescriptor &descriptor);
        /** Deconvolution metadata accessor
         *
         * @return Deconvolution information
         */
        BIPadStrideInfo deconvolution_info() const;
        /** Computes deconvolution output descriptor
         *
         * @param[in] input_descriptor   Input descriptor
         * @param[in] weights_descriptor Weights descriptor
         * @param[in] info               Convolution operation attributes
         *
         * @return Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                            const BITensorDescriptor &weights_descriptor,
                                                            const BIPadStrideInfo    &info);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    private:
        descriptors::DeconvolutionLayerDescriptor descriptor;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_DECONVOLUTIONLAYERNODE_H
