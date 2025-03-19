//
// Created by holynova on 25-3-14.
//

#include "graph/mutators/bi_node_execution_method_mutator.h"

#include "graph/backends/bi_backendRegistry.h"
#include "graph/bi_graph.h"
#include "graph/bi_logger.h"
#include "graph/nodes/Nodes.h"
#include "graph/bi_utils.h"

#include "support/bi_cast.hpp"

namespace BatmanInfer {

namespace graph {

namespace {

    /** Runs a default setter function on a given types of nodes
     *
     * @tparam Setter Setter function to run
     *
     * @param[in, out] g         Graph to extract the nodes from
     * @param[in]      node_type Node type
     * @param[in]      setter    Setter function
     */
    template <typename Setter>
    void set_default_on_invalid_method(BIGraph &g, BINodeType node_type, Setter &&setter)
    {
        const std::vector<NodeID> &node_ids = g.nodes(node_type);
        for (auto &node_id : node_ids)
        {
            BIINode *node = g.node(node_id);
            if (node != nullptr)
            {
                // Validate node
                backends::BIIDeviceBackend &backend =
                    backends::BIBackendRegistry::get().get_backend(node->assigned_target());
                BIStatus                    status  = backend.validate_node(*node);

                // Set default execution method in case of failure
                if (!bool(status))
                {
                    setter(node);
                }
            }
        }
    }

} // namespace

    const char *BINodeExecutionMethodMutator::name()
    {
        return "BINodeExecutionMethodMutator";
    }

    BIIGraphMutator::MutationType BINodeExecutionMethodMutator::type() const
    {
        return BIIGraphMutator::MutationType::Backend;
    }

    void BINodeExecutionMethodMutator::mutate(BIGraph &g)
    {
        // Convolution Layer
        set_default_on_invalid_method(
            g, BINodeType::ConvolutionLayer,
            [](BIINode *n)
            {
                BI_COMPUTE_LOG_GRAPH_INFO("Switched ConvolutionLayer method of node with ID : "
                                          << n->id() << " and Name: " << n->name() << std::endl);
                auto *casted_node = BatmanInfer::utils::cast::polymorphic_downcast<ConvolutionLayerNode *>(n);
                casted_node->set_convolution_method(BIConvolutionMethod::Default);
            });

        // Depthwise Convolution Layer
        set_default_on_invalid_method(
            g, BINodeType::DepthwiseConvolutionLayer,
            [](BIINode *n)
            {
                BI_COMPUTE_LOG_GRAPH_INFO("Switched Depthwise ConvolutionLayer method of node with ID : "
                                          << n->id() << " and Name: " << n->name() << std::endl);
                auto *casted_node = BatmanInfer::utils::cast::polymorphic_downcast<DepthwiseConvolutionLayerNode *>(n);
                casted_node->set_depthwise_convolution_method(BIDepthwiseConvolutionMethod::Default);
            });
    }

} // namespace graph

} // namespace BatmanInfer
