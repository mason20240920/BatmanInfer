//
// Created by holynova on 2025/1/20.
//

#pragma once

#include "graph/bi_types.h"
#include "graph/bi_workload.h"

#include <map>

namespace BatmanInfer {

namespace graph {

    // Forward declaration
    class BIGraph;
    class BIGraphContext;
    class BIPassManager;

    /** 管理 graph 的类
     *
     * Manages a list of graphs along with their resources
     */
    class BIGraphManager final
    {
    public:
        /** Default Constructor **/
        BIGraphManager();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIGraphManager(const BIGraphManager &) = delete;
        /** Default move constructor */
        BIGraphManager(BIGraphManager &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIGraphManager &operator=(const BIGraphManager &) = delete;
        /** Default move assignment operator */
        BIGraphManager &operator=(BIGraphManager &&) = default;
        /** Finalizes a given graph
         *
         * @warning At this given time finalize_graph will alter the passed graph,
         *          plan is to avoid by copying the graph structure,
         *          or provide another entry-point for this functionality as it will increase the memory requirements
         *
         * @param[in] graph  Graph to finalize
         * @param[in] ctx    Graph context
         * @param[in] pm     Pass manager to use for any optimization passes
         * @param[in] target Execution target (Single target execution is currently supported)
         */
        void finalize_graph(BIGraph &graph, BIGraphContext &ctx, BIPassManager &pm, BITarget target);
        /** Executes a graph
         *
         * @param[in] graph Graph to execute
         */
        void execute_graph(BIGraph &graph);
        /** Invalidates the graph execution workload
         *
         * @param[in] graph Graph to invalidate
         */
        void invalidate_graph(BIGraph &graph);

    private:
        std::map<GraphID, BIExecutionWorkload> _workloads = {}; /**< Graph workloads */
    };

} // namespace graph

} // namespace BatmanInfer
