//
// Created by holynova on 2025/1/24.
//

#pragma once

#include <vector>

namespace BatmanInfer {

namespace graph {

    // Forward declarations
    class BIGraph;
    class BIGraphContext;
    struct BIExecutionWorkload;
    class BIITensorHandle;

    namespace detail {

        /** Configures transition manager and execution workload
         *
         * @param[in] g        Graph to configure
         * @param[in] ctx      Graph context
         * @param[in] workload Workload to configure
         */
        void configure_transition_manager(BIGraph &g, BIGraphContext &ctx, BIExecutionWorkload &workload);

    } // namespace detail

} // namespace graph

} // namespace BatmanInfer
