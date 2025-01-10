//
// Created by holynova on 2025/1/9.
//

#include "graph/bi_graph.h"

namespace BatmanInfer {

namespace graph {

    BIGraph::BIGraph(GraphID id, std::string name)
    : _id(id), _name(std::move(name)), _nodes(), _edges(), _tensors(), _tagged_nodes(), _mtx()
    {
    }

    bool BIGraph::remove_node(NodeID nid)
    {
        if (nid >= _nodes.size())
        {
            return false;
        }

        std::unique_ptr<BIINode> &node = _nodes[nid];

        if (node)
        {
            // Remove input connections
            for (auto &input_eid : node->_input_edges)
            {
                remove_connection(input_eid);
            }

            // Remove output connections
            std::set<EdgeID> output_edges_copy = node->output_edges();
            for (auto &output_eid : output_edges_copy)
            {
                remove_connection(output_eid);
            }

            // Remove nid from tagged nodes
            std::vector<NodeID> &tnodes = _tagged_nodes.at(node->type());
            tnodes.erase(std::remove(tnodes.begin(), tnodes.end(), nid), tnodes.end());
        }

        node = nullptr;

        return true;
    }

    EdgeID BIGraph::add_connection(NodeID source, size_t source_idx, NodeID sink, size_t sink_idx)
    {
        BatmanInfer::lock_guard<BatmanInfer::Mutex> lock(_mtx);

        // Check if node index is valid, if node exists and finally if the connection index is valid
        BI_COMPUTE_ERROR_ON((source >= _nodes.size()) || (_nodes[source] == nullptr) ||
                            (source_idx >= _nodes[source]->num_outputs()));
        BI_COMPUTE_ERROR_ON((sink >= _nodes.size()) || (_nodes[sink] == nullptr) ||
                            (sink_idx >= _nodes[sink]->num_inputs()));

        // Get nodes
        std::unique_ptr<BIINode> &source_node = _nodes[source];
        std::unique_ptr<BIINode> &sink_node   = _nodes[sink];

        // Check for duplicate connections (Check only sink node)
        BIEdge *sink_node_edge = sink_node->input_edge(sink_idx);
        if ((sink_node_edge != nullptr) && (sink_node_edge->producer_id() == source) &&
            (sink_node_edge->producer_idx() == source_idx) && (sink_node_edge->consumer_id() == sink) &&
            (sink_node_edge->consumer_idx() == sink_idx))
        {
            return sink_node_edge->id();
        }

        // Check if there is already a tensor associated with output if not create one
        TensorID tid = source_node->output_id(source_idx);
        if (tid == NullTensorID)
        {
            tid = create_tensor();
        }
        std::unique_ptr<BITensor> &tensor = _tensors[tid];

        // Create connections
        EdgeID eid = _edges.size();
        auto   connection =
            std::make_unique<BIEdge>(eid, source_node.get(), source_idx, sink_node.get(), sink_idx, tensor.get());
        _edges.push_back(std::move(connection));

        // Add connections to source and sink nodes
        source_node->_output_edges.insert(eid);
        sink_node->_input_edges[sink_idx] = eid;

        // Set tensor output node
        source_node->_outputs[source_idx] = tid;

        // Bind tensor to the edge
        tensor->bind_edge(eid);

        // Try and propagate shapes in sink node
        sink_node->forward_descriptors();

        return eid;
    }

    bool BIGraph::remove_connection(EdgeID eid)
    {
        if (eid >= _edges.size())
        {
            return false;
        }

        std::unique_ptr<BIEdge> &edge = _edges[eid];

        // Remove node connections
        if (edge != nullptr)
        {
            // Get tensor bound to the edge
            if (edge->tensor() != nullptr)
            {
                edge->tensor()->unbind_edge(eid);
            }

            // Remove edges from source node
            if (edge->producer() != nullptr)
            {
                edge->producer()->_output_edges.erase(eid);
            }

            // Remove edges from sink node
            if ((edge->consumer() != nullptr) && (edge->consumer_idx() < edge->consumer()->_input_edges.size()))
            {
                edge->consumer()->_input_edges[edge->consumer_idx()] = EmptyEdgeID;
            }
        }

        // Clear edge
        edge = nullptr;

        return true;
    }

    TensorID BIGraph::create_tensor(const BITensorDescriptor &desc)
    {
        TensorID tid    = _tensors.size();
        auto     tensor = std::make_unique<BITensor>(tid, desc);
        _tensors.push_back(std::move(tensor));

        return tid;
    }

    std::string BIGraph::name() const
    {
        return _name;
    }

    GraphID BIGraph::id() const
    {
        return _id;
    }

    const std::vector<NodeID> &BIGraph::nodes(BINodeType type)
    {
        return _tagged_nodes[type];
    }

    std::vector<std::unique_ptr<BIINode>> &BIGraph::nodes()
    {
        return _nodes;
    }

    const std::vector<std::unique_ptr<BIINode>> &BIGraph::nodes() const
    {
        return _nodes;
    }

    const std::vector<std::unique_ptr<BIEdge>> &BIGraph::edges() const
    {
        return _edges;
    }

    std::vector<std::unique_ptr<BITensor>> &BIGraph::tensors()
    {
        return _tensors;
    }

    const std::vector<std::unique_ptr<BITensor>> &BIGraph::tensors() const
    {
        return _tensors;
    }

    const BIINode *BIGraph::node(NodeID id) const
    {
        return (id >= _nodes.size()) ? nullptr : _nodes[id].get();
    }

    BIINode *BIGraph::node(NodeID id)
    {
        return (id >= _nodes.size()) ? nullptr : _nodes[id].get();
    }

    const BIEdge *BIGraph::edge(EdgeID id) const
    {
        return (id >= _edges.size()) ? nullptr : _edges[id].get();
    }

    BIEdge *BIGraph::edge(EdgeID id)
    {
        return (id >= _edges.size()) ? nullptr : _edges[id].get();
    }

    const BITensor *BIGraph::tensor(TensorID id) const
    {
        return (id >= _tensors.size()) ? nullptr : _tensors[id].get();
    }

    BITensor *BIGraph::tensor(TensorID id)
    {
        return (id >= _tensors.size()) ? nullptr : _tensors[id].get();
    }

} // namespace graph

} // namespace BatmanInfer
