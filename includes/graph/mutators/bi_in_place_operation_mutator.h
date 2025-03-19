//
// Created by holynova on 25-3-12.
//

#pragma once

#include "graph/bi_igraphMutator.h"

namespace BatmanInfer {

namespace graph {

    /** Mutation pass to optimize operations that can be performed in-place */
    class BIInPlaceOperationMutator final : public BIIGraphMutator
    {
    public:
        // Inherited methods overridden
        void mutate(BIGraph &g) override;
        MutationType type() const override;
        const char  *name() override;
    };

} // namespace graph

} // namespace BatmanInfer
