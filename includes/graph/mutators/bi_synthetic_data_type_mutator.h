//
// Created by holynova on 25-3-12.
//

#pragma once

#include "graph/bi_graph.h"
#include "graph/bi_igraphMutator.h"

namespace BatmanInfer {

namespace graph {

    /** Mutation pass to create synthetic graphs of a given data type */
    class BISyntheticDataTypeMutator final : public BIIGraphMutator
    {
    public:
        // Constructor
        BISyntheticDataTypeMutator(BIDataType mutate_type = BIDataType::QASYMM8);
        // Inherited methods overridden
        void mutate(BIGraph &g) override;
        MutationType type() const override;
        const char  *name() override;

    private:
        BIDataType _mutate_type;
    };

} // namespace graph

} // namespace BatmanInfer
