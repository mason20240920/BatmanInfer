//
// Created by holynova on 2025/2/6.
//

#pragma once

#include "graph/frontend/bi_igraph_front.h"
#include "graph/frontend/bi_types.h"
#include "graph/bi_graph.h"
#include "graph/bi_graphContext.h"
#include "graph/bi_graphManager.h"

namespace BatmanInfer {

namespace graph {

namespace frontend {

    // Forward Declarations
    class BIILayer;

    /* construct graphs */
    class BIGraphFront : public BIIGraphFront {
    public:
        /** Constructor
         *
         * @param[in] id   Stream id
         * @param[in] name Stream name
         */
        BIGraphFront(size_t id, std::string name);
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIGraphFront(const BIGraphFront&) = delete;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIGraphFront& operator=(const BIGraphFront&) = delete;

        /** Finalizes the frontend for an execution target
         *
         * @param[in] target Execution target
         * @param[in] config (Optional) Graph configuration to use
         */
        void finalize(BITarget target, const BIGraphConfig &config);

        void run();

        // Inherited overridden methods
        void           add_layer(BIILayer &layer) override;
        BIGraph       &graph() override;
        const BIGraph &graph() const override;

    private:
        //Important: GraphContext must be declared *before* the GraphManager
        BIGraphContext _ctx;     /**< Graph context to use */
        BIGraphManager _manager; /**< Graph manager */
        BIGraph        _g;       /**< Internal graph representation */

    };

} // namespace frontend

} // namespace graph

} // namespace BatmanInfer
