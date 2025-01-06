//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_FUSEDDEPTHWISECONVOLUTIONBATCHNORMALIZATIONNODE_H
#define BATMANINFER_GRAPH_FUSEDDEPTHWISECONVOLUTIONBATCHNORMALIZATIONNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Fused Depthwise Convolution Batch Normalization node */
    class FusedDepthwiseConvolutionBatchNormalizationNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] epsilon          Epsilon parameter.
         * @param[in] info             Convolution layer attributes.
         * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
         * @param[in] method           (Optional) Convolution method to use
         * @param[in] fused_activation (Optional) Fused activation layer. Disabled if not specified
         */
        FusedDepthwiseConvolutionBatchNormalizationNode(float                        epsilon,
                                                        BIPadStrideInfo              info,
                                                        unsigned int                 depth_multiplier,
                                                        BIDepthwiseConvolutionMethod method,
                                                        BIActivationLayerInfo fused_activation = BIActivationLayerInfo());

        /** Sets the depthwise convolution layer method to use
         *
         * @param[in] method Method to use for depthwise convolution
         */
        void set_depthwise_convolution_method(BIDepthwiseConvolutionMethod method);

        /** Depthwise convolution layer method accessor
         *
         * @note This is an indication on which depthwise convolution layer implementation to use,
         *       if it fails to be created the library's heuristic approach will be used
         *
         * @return Depthwise convolution layer method to be used by the node
         */
        BIDepthwiseConvolutionMethod depthwise_convolution_method() const;

        /** Epsilon parameter accessor
         *
         * @return Epsilon parameter
         */
        float epsilon() const;

        /** Returns fused activation
         *
         * @return Fused activation
         */
        BIActivationLayerInfo fused_activation() const;

        /** Sets fused activation
         *
         * @param[in] fused_activation Fused activation to set
         */
        void set_fused_activation(BIActivationLayerInfo fused_activation);

        /** Computes convolution output descriptor
         *
         * @param[in] input_descriptor   Input descriptor
         * @param[in] weights_descriptor Weights descriptor
         * @param[in] info               Convolution operation attributes
         * @param[in] depth_multiplier   Depth multiplier
         *
         * @return Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                            const BITensorDescriptor &weights_descriptor,
                                                            const BIPadStrideInfo    &info,
                                                            int                     depth_multiplier);

        /** Sets the convolution layer method to use
         *
         * @param[in] method Method to use for convolution
         */
        void set_convolution_method(BIConvolutionMethod method);

        /** Depth multiplier accessor
         *
         * @return Depth multiplier
         */
        unsigned int depth_multiplier() const;

        /** Convolution metadata accessor
         *
         * @return Convolution information
         */
        BIPadStrideInfo convolution_info() const;

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    public:
        static constexpr BINodeType node_type = BINodeType::FusedDepthwiseConvolutionBatchNormalizationLayer;

    private:
        float _epsilon;

        BIPadStrideInfo              _info;
        unsigned int                 _depth_multiplier;
        BIDepthwiseConvolutionMethod _method;
        BIActivationLayerInfo        _fused_activation;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_FUSEDDEPTHWISECONVOLUTIONBATCHNORMALIZATIONNODE_H
