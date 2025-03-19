//
// Created by holynova on 25-3-12.
//

#pragma once

#include "graph/bi_igraphMutator.h"

namespace BatmanInfer {

namespace graph {

    /** Mutation pass to fall-back to default execution method
     *
     * @note This operates on nodes that support multiple execution methods (e.g. ConvolutionLayerNode)
     *       and in case the requested execution method is not supported for a given configuration.
     *       Thus this is a fall-back mechanism to ensure graph execution.
     */
    class BINodeExecutionMethodMutator final : public BIIGraphMutator
    {
    public:
        // Inherited methods overridden
        void mutate(BIGraph &g) override;
        MutationType type() const override;
        const char  *name() override;
    };

} // namespace graph

} // namespace BatmanInfer
