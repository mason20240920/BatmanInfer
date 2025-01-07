//
// Created by holynova on 2025/1/7.
//

#ifndef BATMANINFER_GRAPH_BI_TOPOLOGICALSORT_H
#define BATMANINFER_GRAPH_BI_TOPOLOGICALSORT_H

#include "graph/bi_types.h"

#include <vector>

namespace BatmanInfer {

namespace graph {

    // Forward declarations
    class BIGraph;

    /** Breadth first search traversal
     *
     * @param g Graph to traverse
     *
     * @return A vector with the node id traversal order
     */
    std::vector<NodeID> bfs(BIGraph &g);

    /** Depth first search traversal
     *
     * @param g Graph to traverse
     *
     * @return A vector with the node id traversal order
     */
    std::vector<NodeID> dfs(BIGraph &g);

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_TOPOLOGICALSORT_H
