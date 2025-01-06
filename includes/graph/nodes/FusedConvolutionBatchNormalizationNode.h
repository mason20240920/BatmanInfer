//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_FUSEDCONVOLUTIONBATCHNORMALIZATIONNODE_H
#define BATMANINFER_GRAPH_FUSEDCONVOLUTIONBATCHNORMALIZATIONNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Batch Normalization node */
    class FusedConvolutionBatchNormalizationNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] epsilon          Epsilon parameter.
         * @param[in] info             Convolution layer attributes.
         * @param[in] num_groups       (Optional) Number of groups (Defaults to 1)
         * @param[in] method           (Optional) Convolution method to use
         * @param[in] fast_math_hint   (Optional) Fast math hint
         * @param[in] fused_activation (Optional) Fused activation layer. Disabled if not specified
         */
        FusedConvolutionBatchNormalizationNode(float                 epsilon,
                                               BIPadStrideInfo       info,
                                               unsigned int          num_groups       = 1,
                                               BIConvolutionMethod   method           = BIConvolutionMethod::Default,
                                               BIFastMathHint        fast_math_hint   = BIFastMathHint::Disabled,
                                               BIActivationLayerInfo fused_activation = BIActivationLayerInfo());

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
         *
         * @return Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                            const BITensorDescriptor &weights_descriptor,
                                                            const BIPadStrideInfo    &info);

        /** Sets the convolution layer method to use
         *
         * @param[in] method Method to use for convolution
         */
        void set_convolution_method(BIConvolutionMethod method);

        /** Number of groups in convolution accessor
         *
         * @return Number of groups in convolution
         */
        unsigned int num_groups() const;

        /** Convolution layer method accessor
         *
         * @note This is an indication on which convolution layer implementation to use,
         *       if it fails to be created the library's heuristic approach will be used
         *
         * @return Convolution layer method to be used by the node
         */
        BIConvolutionMethod convolution_method() const;

        /** Sets the fast math hint
         *
         * @param[in] hint Hint to use for convolution
         */
        void set_fast_math_hint(BIFastMathHint hint);

        /** Fast math hint accessor
         *
         * @return Fast math hint to be used by the node
         */
        BIFastMathHint fast_math_hint() const;

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
        static constexpr BINodeType node_type = BINodeType::FusedConvolutionBatchNormalizationLayer;

    private:
        float _epsilon;

        BIPadStrideInfo       _info;
        unsigned int          _num_groups;
        BIConvolutionMethod   _method;
        BIFastMathHint        _fast_math_hint;
        BIActivationLayerInfo _fused_activation;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_FUSEDCONVOLUTIONBATCHNORMALIZATIONNODE_H
