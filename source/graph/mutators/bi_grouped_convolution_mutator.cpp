//
// Created by holynova on 25-3-12.
//

#include "graph/mutators/bi_grouped_convolution_mutator.h"

#include "graph/backends/bi_backendRegistry.h"
#include "graph/bi_graph.h"
#include "graph/bi_graphBuilder.h"
#include "graph/bi_logger.h"
#include "graph/nodes/Nodes.h"
#include "graph/bi_utils.h"

#include "support/bi_cast.hpp"
#include "support/string_support.hpp"

#include <set>

namespace BatmanInfer {

namespace graph {

namespace {

NodeID create_grouped_convolution(BIGraph              &g,
                                  const BINodeParams   &params,
                                  BINodeIdxPair         input,
                                  NodeID                weights,
                                  NodeID                bias,
                                  BIPadStrideInfo       conv_info,
                                  BIConvolutionMethod   method,
                                  BIActivationLayerInfo fused_act,
                                  BIFastMathHint        fast_math_hint,
                                  unsigned int          num_groups)
{
    bool has_bias = (bias != EmptyNodeID);

    // Split input
    const BITensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
    const unsigned int       input_idx   = get_dimension_idx(input_tensor_desc.layout, BIDataLayoutDimension::CHANNEL);
    NodeID                   input_split = GraphBuilder::add_split_node(g, params, input, num_groups, input_idx);

    // Split weights
    const BITensorDescriptor weights_tensor_desc = get_tensor_descriptor(g, g.node(weights)->outputs()[0]);
    const unsigned int       batch_idx     = get_dimension_idx(weights_tensor_desc.layout, BIDataLayoutDimension::BATCHES);
    NodeID                   weights_split = GraphBuilder::add_split_node(g, params, {weights, 0}, num_groups, batch_idx);

    // Split bias
    NodeID bias_split = EmptyNodeID;
    if (has_bias)
    {
        // Split bias
        bias_split = GraphBuilder::add_split_node(g, params, {bias, 0}, num_groups, 0);
    }

    std::vector<BINodeIdxPair> convolution_outputs;
    for (unsigned int i = 0; i < num_groups; ++i)
    {
        BINodeParams group_params = params;
        NodeID       conv_nid     = g.add_node<ConvolutionLayerNode>(conv_info, 1, method, fast_math_hint);
        g.add_connection(input_split, i, conv_nid, 0);
        g.add_connection(weights_split, i, conv_nid, 1);
        if (has_bias)
        {
            g.add_connection(bias_split, i, conv_nid, 2);
        }

        // Add group name
        if (!group_params.name.empty())
        {
            group_params.name.append("_g" + BatmanInfer::support::cpp11::to_string(i));
        }

        // Set node parameters
        BIINode *node = g.node(conv_nid);
        BI_COMPUTE_ERROR_ON(node == nullptr);
        node->set_common_node_parameters(group_params);

        // Down-cast node
        auto *conv_node = BatmanInfer::utils::cast::polymorphic_downcast<ConvolutionLayerNode *>(node);
        conv_node->set_fused_activation(fused_act);

        convolution_outputs.push_back({conv_nid, 0});
    }

    // Depth concatenate output
    return GraphBuilder::add_concatenate_node(g, params, convolution_outputs, BIDataLayoutDimension::CHANNEL);
}

} // namespace

    const char *BIGroupedConvolutionMutator::name()
    {
        return "BIGroupedConvolutionMutator";
    }

    BIIGraphMutator::MutationType BIGroupedConvolutionMutator::type() const
    {
        return BIIGraphMutator::MutationType::Backend;
    }

    void BIGroupedConvolutionMutator::mutate(BIGraph &g)
    {
        // Early exit if no Convolution layers exist in graph
        if (g.nodes(BINodeType::ConvolutionLayer).empty())
        {
            return;
        }

        // Total nodes
        size_t total_nodes = g.nodes().size();

        // Iterate over convolution nodes
        for (unsigned int i = 0; i < total_nodes; ++i)
        {
            BIINode *node = g.node(i);
            if (node != nullptr && node->type() == BINodeType::ConvolutionLayer &&
                BatmanInfer::utils::cast::polymorphic_downcast<ConvolutionLayerNode *>(node)->num_groups() != 1)
            {
                // Validate node
                backends::BIIDeviceBackend &backend = backends::BIBackendRegistry::get().get_backend(node->assigned_target());
                BIStatus                    status  = backend.validate_node(*node);

                // If grouped convolution is not supported
                if (!bool(status))
                {
                    // Down-cast node
                    auto *conv_node = BatmanInfer::utils::cast::polymorphic_downcast<ConvolutionLayerNode *>(node);

                    // Get internal convolution info
                    const BIPadStrideInfo       conv_info       = conv_node->convolution_info();
                    const BIConvolutionMethod   conv_method     = conv_node->convolution_method();
                    const BIActivationLayerInfo fused_act_info  = conv_node->fused_activation();
                    const BIFastMathHint        fast_math_hint  = conv_node->fast_math_hint();
                    const unsigned int          num_groups      = conv_node->num_groups();
                    const BINodeParams          params          = conv_node->common_node_params();
                    const BITarget              assigned_target = conv_node->assigned_target();

                    // Extract node ids
                    BI_COMPUTE_ERROR_ON(conv_node->input_edge(0) == nullptr || conv_node->input_edge(1) == nullptr);
                    const NodeID input_id   = conv_node->input_edge(0)->producer()->id();
                    const NodeID weights_id = conv_node->input_edge(1)->producer()->id();
                    const NodeID bias_id =
                        (conv_node->input_edge(2) != nullptr) ? conv_node->input_edge(2)->producer()->id() : EmptyNodeID;

                    // Get driving nodes
                    std::vector<BINodeIdxPair> driving_nodes = get_driving_nodes(*node);

                    // Extract activation node accessor if any
                    auto node_accessor = conv_node->output(0)->extract_accessor();

                    // Current max tensor and node id
                    TensorID latest_tid = g.tensors().size();
                    NodeID   latest_nid = g.nodes().size();

                    // Create grouped convolution node
                    NodeID grouped_conv_id =
                        create_grouped_convolution(g, params, {input_id, 0}, weights_id, bias_id, conv_info, conv_method,
                                                   fused_act_info, fast_math_hint, num_groups);

                    // Remove convolution node
                    g.remove_node(node->id());

                    // Update batch normalization node outputs
                    for (auto &driving_node : driving_nodes)
                    {
                        g.add_connection(grouped_conv_id, 0, driving_node.node_id, driving_node.index);
                    }

                    // Update accessor to batch normalization node
                    g.node(grouped_conv_id)->output(0)->set_accessor(std::move(node_accessor));

                    // Configure new tensors and nodes
                    std::for_each(g.tensors().begin() + latest_tid, g.tensors().end(),
                                  [](std::unique_ptr<BITensor> &t) { configure_tensor(t.get()); });
                    std::for_each(g.nodes().begin() + latest_nid, g.nodes().end(),
                                  [&assigned_target](std::unique_ptr<BIINode> &n)
                                  {
                                      if (n != nullptr)
                                      {
                                          n->set_assigned_target(assigned_target);
                                      }
                                  });
                }
            }
        }
    }

} // namespace graph

} // namespace BatmanInfer
