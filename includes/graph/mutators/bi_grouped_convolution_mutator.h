//
// Created by holynova on 25-3-12.
//

#pragma once

#include "graph/bi_igraphMutator.h"

namespace BatmanInfer {

namespace graph {

    /** Mutation pass to implement/optimize grouped convolutions
     *
     * @warning This is compulsory to run in case of grouped convolutions
     **/
    class BIGroupedConvolutionMutator final : public BIIGraphMutator
    {
    public:
        // Inherited methods overridden
        void mutate(BIGraph &g) override;
        MutationType type() const override;
        const char  *name() override;
    };

} // namepsace graph

} // namespace BatmanInfer
