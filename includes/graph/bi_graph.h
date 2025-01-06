//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_BI_GRAPH_H
#define BATMANINFER_GRAPH_BI_GRAPH_H

#include "graph/bi_edge.h"
#include "graph/bi_inode.h"
#include "graph/bi_tensor.h"
#include "graph/bi_types.h"
#include "support/mutex.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef BARE_METAL
#include <thread>
#endif /* BARE_METAL */

namespace BatmanInfer {

namespace graph {

    /** Graph class
     *
     * Represents a multiple source - multiple sink directed graph
     */
    class BIGraph final
    {
    public:
        BIGraph() = default;
        /** Constructor
         *
         * @param[in] id   Graph identification number. Can be used to differentiate between graphs. Default value 0
         * @param[in] name Graph name. Default value empty string
         */
        BIGraph(GraphID id, std::string name);
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIGraph(const BIGraph &) = delete;
        /** Prevent instances of this class from being copy assigned (As this class contains pointers) */
        BIGraph &operator=(const BIGraph &) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BIGraph(BIGraph &&) = delete;
        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BIGraph &operator=(BIGraph &&) = delete;
        /** Adds a node to the graph
         *
         * @note Models a single output node
         *
         * @tparam NT Node operation
         * @tparam Ts Arguments to operation
         *
         * @param[in] args Node arguments
         *
         * @return ID of the node
         */
        template <typename NT, typename... Ts>
        NodeID add_node(Ts &&...args);
        /** Remove the node with the given ID
         *
         * @param[in] nid ID of the node to remove
         *
         * @return True if the removal took place else false
         */
        bool remove_node(NodeID nid);
        /** Adds a connection between two nodes
         *
         * @param[in] source     ID of the source node
         * @param[in] source_idx Output index of the source node
         * @param[in] sink       ID of the sink node
         * @param[in] sink_idx   Input index of the sink node
         *
         * @return ID of this connection
         */
        EdgeID add_connection(NodeID source, size_t source_idx, NodeID sink, size_t sink_idx);
        /** Removes an edge (connection)
         *
         * @param[in] eid Connection to remove
         *
         * @return True if the removal took place else false
         */
        bool remove_connection(EdgeID eid);
        /** Returns graph name
         *
         * @return Graph name
         */
        std::string name() const;
        /** Returns graph id
         *
         * @return Graph id
         */
        GraphID id() const;
        /** Returns graph input nodes
         *
         * @param[in] type Type of nodes to return
         *
         * @return vector containing the graph node of given type
         */
        const std::vector<NodeID> &nodes(BINodeType type);
        /** Returns nodes of graph
         *
         * @warning Nodes can be nullptr if they have been removed during the mutation steps of the graph
         *
         * @return Nodes of graph
         */
        std::vector<std::unique_ptr<BIINode>> &nodes();
        /** Returns nodes of graph
         *
         * @warning Nodes can be nullptr if they have been removed during the mutation steps of the graph
         *
         * @return Nodes of graph
         */
        const std::vector<std::unique_ptr<BIINode>> &nodes() const;
        /** Returns edges of graph
         *
         * @warning Edges can be nullptr if they have been removed during the mutation steps of the graph
         *
         * @return Edges of graph
         */
        const std::vector<std::unique_ptr<BIEdge>> &edges() const;
        /** Returns tensors of graph
         *
         * @warning Tensor can be nullptr if they have been removed during the mutation steps of the graph
         *
         * @return Tensors of graph
         */
        std::vector<std::unique_ptr<BITensor>> &tensors();
        /** Returns tensors of graph
         *
         * @warning Tensor can be nullptr if they have been removed during the mutation steps of the graph
         *
         * @return Tensors of graph
         */
        const std::vector<std::unique_ptr<BITensor>> &tensors() const;
        /** Get node object given its id
         *
         * @warning Can be nullptr if node was removed during the mutation steps of the graph
         *
         * @param[in] id Node ID
         *
         * @return The actual node object
         */
        const BIINode *node(NodeID id) const;
        /** Get node object given its id
         *
         * @warning Can be nullptr if node was removed during the mutation steps of the graph
         *
         * @param[in] id Node ID
         *
         * @return The actual node object
         */
        BIINode *node(NodeID id);
        /** Get edge object given its id
         *
         * @warning Can be nullptr if node was removed during the mutation steps of the graph
         *
         * @param[in] id Edge ID
         *
         * @return The actual edge object
         */
        const BIEdge *edge(EdgeID id) const;
        /** Get edge object given its id
         *
         * @warning Can be nullptr if node was removed during the mutation steps of the graph
         *
         * @param[in] id Edge ID
         *
         * @return The actual edge object
         */
        BIEdge *edge(EdgeID id);
        /** Get tensor object given its id
         *
         * @warning Can be nullptr if tensor was removed during the mutation steps of the graph
         *
         * @param[in] id Tensor ID
         *
         * @return The actual tensor object
         */
        const BITensor *tensor(TensorID id) const;
        /** Get tensor object given its id
         *
         * @warning Can be nullptr if tensor was removed during the mutation steps of the graph
         *
         * @param[in] id Tensor ID
         *
         * @return The actual tensor object
         */
        BITensor *tensor(TensorID id);

    private:
        /** Creates a tensor object
         *
         * @param[in] desc Tensor descriptor
         *
         * @return Tensor ID
         */
        TensorID create_tensor(const BITensorDescriptor &desc = BITensorDescriptor());

    private:
        GraphID                                   _id           = GraphID(0); /**< Graph id */
        std::string                               _name         = {};         /**< Graph name */
        std::vector<std::unique_ptr<BIINode>>     _nodes        = {};         /**< Graph nodes */
        std::vector<std::unique_ptr<BIEdge>>      _edges        = {};         /**< Graph edges */
        std::vector<std::unique_ptr<BITensor>>    _tensors      = {};         /**< Graph tensors */
        std::map<BINodeType, std::vector<NodeID>> _tagged_nodes = {}; /**< Graph nodes map with the node type as key */
        BatmanInfer::Mutex                        _mtx          = {}; /**< Mutex used for graph construction */
    };

    template <typename NT, typename... Ts>
    inline NodeID BIGraph::add_node(Ts &&...args)
    {
        BatmanInfer::lock_guard<BatmanInfer::Mutex> lock(_mtx);

        // Create node
        NodeID nid  = _nodes.size();
        auto   node = std::make_unique<NT>(std::forward<Ts>(args)...);
        node->set_graph(this);
        node->set_id(nid);

        // Keep track of input nodes
        _tagged_nodes[node->type()].push_back(nid);

        // Associate a new tensor with each output
        for (auto &output : node->_outputs)
        {
            output = create_tensor();
        }

        // Propagate node shape if possible
        node->forward_descriptors();

        // Add node to the graph nodes
        _nodes.push_back(std::move(node));

        return nid;
    }

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_GRAPH_H
