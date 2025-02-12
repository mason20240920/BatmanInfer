//
// Created by holynova on 2025/1/20.
//

#include "graph/bi_graphManager.h"

#include "graph/algorithms/bi_topologicalSort.h"
#include "graph/detail/bi_cross_layer_memory_manager_helpers.h"
#include "graph/detail/bi_execution_helpers.h"
#include "graph/bi_graph.h"
#include "graph/bi_graphContext.h"
#include "graph/bi_logger.h"
#include "graph/bi_passManager.h"
#include "graph/bi_typePrinter.h"
#include "graph/bi_utils.h"
#include "common/utils/bi_log.hpp"

namespace BatmanInfer {

namespace graph {

    BIGraphManager::BIGraphManager() : _workloads()
    {
    }

    void BIGraphManager::finalize_graph(BIGraph &graph, BIGraphContext &ctx, BIPassManager &pm, BITarget target)
    {
        BI_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Initiate graph configuration!");

        // Check if graph has been registered
        if (_workloads.find(graph.id()) != std::end(_workloads))
        {
            BI_COMPUTE_ERROR("Graph is already registered!");
        }

        // Apply IR mutating passes
        pm.run_type(graph, BIIGraphMutator::MutationType::IR);

        // Force target to all graph construct
        BITarget forced_target = target;

        // In case CLVK is selected, use the CL backend and
        // update config
        // if (target == BITarget::CLVK)
        // {
        //     forced_target        = BITarget::CL;
        //     BIGraphConfig config = ctx.config();
        //     // config.backend_type = CLBackendType::Clvk;
        //     ctx.set_config(config);
        // }

        if (!is_target_supported(target))
        {
            forced_target = get_default_target();
            BI_COMPUTE_LOG_GRAPH_INFO("Switching target from " << target << " to " << forced_target << std::endl);
        }
        force_target_to_graph(graph, forced_target);

        // Setup backend context
        setup_requested_backend_context(ctx, forced_target);

        // Configure all tensors
        detail::configure_all_tensors(graph);

        // Apply backend mutating passes
        pm.run_type(graph, BIIGraphMutator::MutationType::Backend);

        // Perform topological sort
        std::vector<NodeID> topological_sorted_nodes = dfs(graph);

        // Validate all nodes
        detail::validate_all_nodes(graph);

        // Configure all nodes
        auto workload = detail::configure_all_nodes(graph, ctx, topological_sorted_nodes);
        BI_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");

        // Allocate const tensors and call accessors
        detail::allocate_const_tensors(graph);
        detail::call_all_const_node_accessors(graph);

        // Prepare graph
        detail::prepare_all_tasks(workload);

        // Setup tensor memory (Allocate all tensors or setup transition manager)
        if (ctx.config().use_transition_memory_manager)
        {
            detail::configure_transition_manager(graph, ctx, workload);
        }
        else
        {
            detail::allocate_all_tensors(graph);
        }

        // Finalize Graph context
        ctx.finalize();

        // Register graph
        _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
        BI_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id() << std::endl);
    }

    void BIGraphManager::execute_graph(BIGraph &graph)
    {
        BI_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Initiate graph execution!");

        // Check if graph is finalized
        auto it = _workloads.find(graph.id());
        BI_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

        while (true)
        {
            // Call input accessors
            if (!detail::call_all_input_node_accessors(it->second))
            {
                return;
            }

            // Run graph
            detail::call_all_tasks(it->second);

            // Call output accessors
            if (!detail::call_all_output_node_accessors(it->second))
            {
                return;
            }
        }
    }

    void BIGraphManager::invalidate_graph(BIGraph &graph)
    {
        auto it = _workloads.find(graph.id());
        BI_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

        _workloads.erase(it);
    }

} // namespace graph

} // namespace BatmanInfer
