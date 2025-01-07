//
// Created by holynova on 2025/1/7.
//

#include "graph/algorithms/bi_topologicalSort.h"
#include "graph/bi_graph.h"
#include "support/bi_iterable.h"

#include <list>
#include <stack>

namespace BatmanInfer {

namespace graph {

    namespace detail {

        /** Checks if all the input dependencies of a node have been visited
         *
         * @param[in] node    Node to check
         * @param[in] visited Vector that contains the visited information
         *
         * @return True if all inputs dependencies have been visited else false
         */
        inline bool all_inputs_are_visited(const BIINode *node, const std::vector<bool> &visited)
        {
            BI_COMPUTE_ERROR_ON(node == nullptr);
            const BIGraph *graph = node->graph();
            BI_COMPUTE_ERROR_ON(graph == nullptr);

            bool are_all_visited = true;
            for (const auto &input_edge_id : node->input_edges())
            {
                if (input_edge_id != EmptyNodeID)
                {
                    const BIEdge *input_edge = graph->edge(input_edge_id);
                    BI_COMPUTE_ERROR_ON(input_edge == nullptr);
                    BI_COMPUTE_ERROR_ON(input_edge->producer() == nullptr);
                    if (!visited[input_edge->producer_id()])
                    {
                        are_all_visited = false;
                        break;
                    }
                }
            }

            return are_all_visited;
        }

    } // namespace detail

    std::vector<NodeID> bfs(BIGraph &g)
    {
        std::vector<NodeID> bfs_order_vector;

        // Created visited vector
        std::vector<bool> visited(g.nodes().size(), false);

        // Create BFS queue
        std::list<NodeID> queue;

        // Push inputs and mark as visited
        for (auto &input : g.nodes(BINodeType::Input))
        {
            if (input != EmptyNodeID)
            {
                visited[input] = true;
                queue.push_back(input);
            }
        }

        // Push const nodes and mark as visited
        for (auto &const_node : g.nodes(BINodeType::Const))
        {
            if (const_node != EmptyNodeID)
            {
                visited[const_node] = true;
                queue.push_back(const_node);
            }
        }

        // Iterate over vector and edges
        while (!queue.empty())
        {
            // Dequeue a node from queue and process
            NodeID n = queue.front();
            bfs_order_vector.push_back(n);
            queue.pop_front();

            const BIINode *node = g.node(n);
            BI_COMPUTE_ERROR_ON(node == nullptr);
            for (const auto &eid : node->output_edges())
            {
                const BIEdge *e = g.edge(eid);
                BI_COMPUTE_ERROR_ON(e == nullptr);
                if (!visited[e->consumer_id()] && detail::all_inputs_are_visited(e->consumer(), visited))
                {
                    visited[e->consumer_id()] = true;
                    queue.push_back(e->consumer_id());
                }
            }
        }

        return bfs_order_vector;
    }

    std::vector<NodeID> dfs(BIGraph &g)
    {
        std::vector<NodeID> dfs_order_vector;

        // Created visited vector
        std::vector<bool> visited(g.nodes().size(), false);

        // Create DFS stack
        std::stack<NodeID> stack;

        // Push inputs and mark as visited
        for (auto &input : g.nodes(BINodeType::Input))
        {
            if (input != EmptyNodeID)
            {
                visited[input] = true;
                stack.push(input);
            }
        }

        // Push const nodes and mark as visited
        for (auto &const_node : g.nodes(BINodeType::Const))
        {
            if (const_node != EmptyNodeID)
            {
                visited[const_node] = true;
                stack.push(const_node);
            }
        }

        // Iterate over vector and edges
        while (!stack.empty())
        {
            // Pop a node from stack and process
            NodeID n = stack.top();
            dfs_order_vector.push_back(n);
            stack.pop();

            // Mark node as visited
            if (!visited[n])
            {
                visited[n] = true;
            }

            const BIINode *node = g.node(n);
            BI_COMPUTE_ERROR_ON(node == nullptr);
            // Reverse iterate to push branches from right to left and pop on the opposite order
            for (const auto &eid : BatmanInfer::utils::iterable::reverse_iterate(node->output_edges()))
            {
                const BIEdge *e = g.edge(eid);
                BI_COMPUTE_ERROR_ON(e == nullptr);
                if (!visited[e->consumer_id()] && detail::all_inputs_are_visited(e->consumer(), visited))
                {
                    stack.push(e->consumer_id());
                }
            }
        }

        return dfs_order_vector;
    }

} // namespace graph

} // namespace BatmanInfer


