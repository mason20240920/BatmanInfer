//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_BI_INODEVISITOR_H
#define BATMANINFER_GRAPH_BI_INODEVISITOR_H

#include "graph/nodes/NodesFwd.h"

namespace BatmanInfer {

namespace graph {

    /**  Node visitor interface */
    class BIINodeVisitor
    {
    public:
        /** Default destructor. */
        virtual ~BIINodeVisitor() = default;
        /** Visit INode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(BIINode &n) = 0;
        /** Visit ActivationLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(ActivationLayerNode &n) = 0;
        /** Visit BatchNormalizationLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(BatchNormalizationLayerNode &n) = 0;
        /** Visit ConcatenateLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(ConcatenateLayerNode &n) = 0;
        /** Visit ConstNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(ConstNode &n) = 0;
        /** Visit ConvolutionLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(ConvolutionLayerNode &n) = 0;
        /** Visit DepthwiseConvolutionLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(DepthwiseConvolutionLayerNode &n) = 0;
        /** Visit DequantizationLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(DequantizationLayerNode &n) = 0;
        /** Visit DetectionOutputLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(DetectionOutputLayerNode &n) = 0;
        /** Visit DetectionPostProcessLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(DetectionPostProcessLayerNode &n) = 0;
        /** Visit EltwiseLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(EltwiseLayerNode &n) = 0;
        /** Visit FlattenLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(FlattenLayerNode &n) = 0;
        /** Visit FullyConnectedLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(FullyConnectedLayerNode &n) = 0;
        /** Visit FusedConvolutionBatchNormalizationNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(FusedConvolutionBatchNormalizationNode &n) = 0;
        /** Visit FusedDepthwiseConvolutionBatchNormalizationNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(FusedDepthwiseConvolutionBatchNormalizationNode &n) = 0;
        /** Visit InputNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(InputNode &n) = 0;
        /** Visit NormalizationLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(NormalizationLayerNode &n) = 0;
        /** Visit OutputNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(OutputNode &n) = 0;
        /** Visit PermuteLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(PermuteLayerNode &n) = 0;
        /** Visit PreluLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(PReluLayerNode &n) = 0;
        /** Visit PoolingLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(PoolingLayerNode &n) = 0;
        /** Visit PrintLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(PrintLayerNode &n) = 0;
        /** Visit PriorBoxLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(PriorBoxLayerNode &n) = 0;
        /** Visit QuantizationLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(QuantizationLayerNode &n) = 0;
        /** Visit ReshapeLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(ReshapeLayerNode &n) = 0;
        /** Visit SoftmaxLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(SoftmaxLayerNode &n) = 0;
        /** Visit SplitLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(SplitLayerNode &n) = 0;
        /** Visit StackLayerNode.
         *
         * @param[in] n Node to visit.
         */
        virtual void visit(StackLayerNode &n) = 0;
    };

    /** Default visitor implementation
     *
     * Implements visit methods by calling a default function.
     * Inherit from DefaultNodeVisitor if you don't want to provide specific implementation for all nodes.
     */
    class DefaultNodeVisitor : public BIINodeVisitor
    {
    public:
        /** Default destructor */
        virtual ~DefaultNodeVisitor() = default;

    #ifndef DOXYGEN_SKIP_THIS
        // Inherited methods overridden
        virtual void visit(BIINode &n) override;
        virtual void visit(ActivationLayerNode &n) override;
        virtual void visit(BatchNormalizationLayerNode &n) override;
        virtual void visit(ConcatenateLayerNode &n) override;
        virtual void visit(ConstNode &n) override;
        virtual void visit(ConvolutionLayerNode &n) override;
        virtual void visit(DequantizationLayerNode &n) override;
        virtual void visit(DetectionOutputLayerNode &n) override;
        virtual void visit(DetectionPostProcessLayerNode &n) override;
        virtual void visit(DepthwiseConvolutionLayerNode &n) override;
        virtual void visit(EltwiseLayerNode &n) override;
        virtual void visit(FlattenLayerNode &n) override;
        virtual void visit(FullyConnectedLayerNode &n) override;
        virtual void visit(FusedConvolutionBatchNormalizationNode &n) override;
        virtual void visit(FusedDepthwiseConvolutionBatchNormalizationNode &n) override;
        virtual void visit(InputNode &n) override;
        virtual void visit(NormalizationLayerNode &n) override;
        virtual void visit(OutputNode &n) override;
        virtual void visit(PermuteLayerNode &n) override;
        virtual void visit(PoolingLayerNode &n) override;
        virtual void visit(PReluLayerNode &n) override;
        virtual void visit(PrintLayerNode &n) override;
        virtual void visit(PriorBoxLayerNode &n) override;
        virtual void visit(QuantizationLayerNode &n) override;
        virtual void visit(ReshapeLayerNode &n) override;
        virtual void visit(SoftmaxLayerNode &n) override;
        virtual void visit(SplitLayerNode &n) override;
        virtual void visit(StackLayerNode &n) override;
    #endif /* DOXYGEN_SKIP_THIS */

        /** Function to be overloaded by the client and implement default behavior for the
         *  non-overloaded visitors
         */
        virtual void default_visit(BIINode &n) = 0;
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_INODEVISITOR_H
