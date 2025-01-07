//
// Created by holynova on 2025/1/7.
//

#include "graph/bi_inode.h"
#include "data/core/bi_error.h"
#include "graph/bi_edge.h"
#include "graph/bi_graph.h"
#include "graph/bi_tensor.h"

namespace BatmanInfer {

namespace graph {

    BIINode::BIINode()
    : _graph(nullptr), _id(EmptyNodeID), _common_params({ "", BITarget::UNSPECIFIED}),
      _outputs(), _input_edges(), _output_edges(), _assigned_target(BITarget::UNSPECIFIED)
    {
    }

    BIStatus BIINode::validate() const
    {
        return BIStatus{};
    }

    void BIINode::set_graph(BIGraph *g)
    {
        BI_COMPUTE_ERROR_ON(g == nullptr);
        _graph = g;
    }

    void BIINode::set_id(NodeID id)
    {
        _id = id;
    }

    void BIINode::set_common_node_parameters(BINodeParams common_params)
    {
        _common_params = std::move(common_params);
    }

    void BIINode::set_requested_target(BITarget target)
    {
        _common_params.target = target;
    }

    void BIINode::set_assigned_target(BITarget target)
    {
        _assigned_target = target;
    }

    void BIINode::set_output_tensor(TensorID tid, size_t idx)
    {
        BI_COMPUTE_ERROR_ON(_graph == nullptr);
        if (tid != NullTensorID && (idx < _outputs.size()) && (_graph->tensor(tid) != nullptr))
        {
            BITensor *updated_tensor = _graph->tensor(tid);
            _outputs[idx]          = tid;

            // Set tensor to all output edges of the node
            for (auto &output_edge_id : _output_edges)
            {
                auto output_edge = _graph->edge(output_edge_id);
                if (output_edge != nullptr)
                {
                    // Unbind edge from current tensor
                    auto current_output_tensor = output_edge->tensor();
                    current_output_tensor->unbind_edge(output_edge->id());

                    // Update tensor to edge and rebind tensor
                    output_edge->update_bound_tensor(updated_tensor);
                    updated_tensor->bind_edge(output_edge->id());
                }
            }
        }
    }

    NodeID BIINode::id() const
    {
        return _id;
    }

    std::string BIINode::name() const
    {
        return _common_params.name;
    }

    const BIGraph *BIINode::graph() const
    {
        return _graph;
    }

    BIGraph *BIINode::graph()
    {
        return _graph;
    }

    const std::vector<TensorID> &BIINode::outputs() const
    {
        return _outputs;
    }

    const std::vector<EdgeID> &BIINode::input_edges() const
    {
        return _input_edges;
    }

    const std::set<EdgeID> &BIINode::output_edges() const
    {
        return _output_edges;
    }

    TensorID BIINode::input_id(size_t idx) const
    {
        BI_COMPUTE_ERROR_ON(_graph == nullptr);
        BI_COMPUTE_ERROR_ON(idx >= _input_edges.size());
        BIEdge *e = _graph->edge(_input_edges[idx]);
        return (e != nullptr) ? e->tensor_id() : NullTensorID;
    }

    TensorID BIINode::output_id(size_t idx) const
    {
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());
        return _outputs[idx];
    }

    BITensor *BIINode::input(size_t idx) const
    {
        BI_COMPUTE_ERROR_ON(_graph == nullptr);
        BI_COMPUTE_ERROR_ON(idx >= _input_edges.size());
        BIEdge *e = _graph->edge(_input_edges[idx]);
        return (e != nullptr) ? e->tensor() : nullptr;
    }

    BITensor *BIINode::output(size_t idx) const
    {
        BI_COMPUTE_ERROR_ON(_graph == nullptr);
        BI_COMPUTE_ERROR_ON(idx >= _outputs.size());
        return _graph->tensor(_outputs[idx]);
    }

    EdgeID BIINode::input_edge_id(size_t idx) const
    {
        BI_COMPUTE_ERROR_ON(idx >= _input_edges.size());
        return _input_edges[idx];
    }

    BIEdge *BIINode::input_edge(size_t idx) const
    {
        BI_COMPUTE_ERROR_ON(_graph == nullptr);
        BI_COMPUTE_ERROR_ON(idx >= _input_edges.size());
        return _graph->edge(_input_edges[idx]);
    }

    size_t BIINode::num_inputs() const
    {
        return _input_edges.size();
    }

    size_t BIINode::num_outputs() const
    {
        return _outputs.size();
    }

    BINodeParams BIINode::common_node_params() const
    {
        return _common_params;
    }

    BITarget BIINode::requested_target() const
    {
        return _common_params.target;
    }

    BITarget BIINode::assigned_target() const
    {
        return _assigned_target;
    }

} // namespace graph

} // namespace BatmanInfer


