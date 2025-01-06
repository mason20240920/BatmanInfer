//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_GRAPH_DEPTHWISECONVOLUTIONLAYERNODE_H
#define BATMANINFER_GRAPH_DEPTHWISECONVOLUTIONLAYERNODE_H

#include "graph/bi_inode.h"

namespace BatmanInfer {

namespace graph {

    /** Depthwise Convolution Layer node */
    class DepthwiseConvolutionLayerNode final : public BIINode
    {
    public:
        /** Constructor
         *
         * @param[in] info             Convolution layer attributes
         * @param[in] depth_multiplier (Optional) Depth multiplier parameter.
         * @param[in] method           (Optional) Depthwise convolution method to use
         * @param[in] out_quant_info   (Optional) Output quantization info
         */
        DepthwiseConvolutionLayerNode(BIPadStrideInfo              info,
                                      int                          depth_multiplier = 1,
                                      BIDepthwiseConvolutionMethod method           = BIDepthwiseConvolutionMethod::Default,
                                      BIQuantizationInfo           out_quant_info   = BIQuantizationInfo());
        /** Sets the depthwise convolution method to use
         *
         * @param[in] method Depthwise convolution method to use
         */
        void set_depthwise_convolution_method(BIDepthwiseConvolutionMethod method);
        /** Depthwise convolution layer method accessor
         *
         * @note This is an indication on which depthwise implementation to use,
         *       if it fails to be created the generic approach will be used
         *
         * @return Depthwise convolution layer method do be used by the node
         */
        BIDepthwiseConvolutionMethod depthwise_convolution_method() const;
        /** Depth multiplier accessor
         *
         * @return Depth multiplier
         */
        int depth_multiplier() const;
        /** Convolution metadata accessor
         *
         * @return Convolution information
         */
        BIPadStrideInfo convolution_info() const;
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
        /** Sets convolution info
         *
         * @param[in] info Convolution info to set
         */
        void set_convolution_info(BIPadStrideInfo info);
        /** Computes depthwise convolution output descriptor
         *
         * @param[in] input_descriptor   Input descriptor
         * @param[in] weights_descriptor Weights descriptor
         * @param[in] info               Convolution operation attributes
         * @param[in] depth_multiplier   (Optional) Depth multiplier parameter.
         *
         * @return Output descriptor
         */
        static BITensorDescriptor compute_output_descriptor(const BITensorDescriptor &input_descriptor,
                                                            const BITensorDescriptor &weights_descriptor,
                                                            const BIPadStrideInfo    &info,
                                                            int                     depth_multiplier = 1);

        // Inherited overridden methods:
        BINodeType         type() const override;
        bool               forward_descriptors() override;
        BITensorDescriptor configure_output(size_t idx) const override;
        void               accept(BIINodeVisitor &v) override;

    public:
        static constexpr BINodeType node_type = BINodeType::DepthwiseConvolutionLayer;

    private:
        BIPadStrideInfo              _info;
        int                          _depth_multiplier;
        BIDepthwiseConvolutionMethod _method;
        BIQuantizationInfo           _out_quant_info;
        BIActivationLayerInfo        _fused_activation;
    };

} // namespace graph

} // namespace graph

#endif //BATMANINFER_GRAPH_DEPTHWISECONVOLUTIONLAYERNODE_H
