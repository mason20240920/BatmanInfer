//
// Created by holynova on 25-3-12.
//

#pragma once

#include "graph/bi_graph.h"
#include "graph/bi_igraphMutator.h"

namespace BatmanInfer {

namespace graph {

    /** Mutation pass to fuss nodes */
    class BINodeFusionMutator final : public BIIGraphMutator
    {
    public:
        // Inherited methods overridden
        void mutate(BIGraph &g) override;
        MutationType type() const override;
        const char  *name() override;
    };

}// namespace graph

} // namespace BatmanInfer
