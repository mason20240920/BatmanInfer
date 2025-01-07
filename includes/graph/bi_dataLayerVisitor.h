//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_BI_DATALAYERVISITOR_H
#define BATMANINFER_GRAPH_BI_DATALAYERVISITOR_H

#include "graph/bi_igraphPrinter.h"
#include "graph/bi_inodeVisitor.h"
#include "graph/bi_types.h"

namespace BatmanInfer {

namespace graph {

    /** Graph printer visitor. */
    class DataLayerVisitor final : public DefaultNodeVisitor
    {
    public:
        using LayerData = std::map<std::string, std::string>;
        /** Default Constructor **/
        DataLayerVisitor() = default;

        const LayerData &layer_data() const;

        // Reveal Parent method
        using DefaultNodeVisitor::visit;
        // Inherited methods overridden
        void visit(ConvolutionLayerNode &n) override;
        void visit(DepthwiseConvolutionLayerNode &n) override;
        void visit(FusedConvolutionBatchNormalizationNode &n) override;
        void visit(FusedDepthwiseConvolutionBatchNormalizationNode &n) override;
        void visit(OutputNode &n) override;

        void default_visit(BIINode &n) override;

    private:
        LayerData _layer_data{};
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_DATALAYERVISITOR_H
