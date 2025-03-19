//
// Created by holynova on 25-3-14.
//

#include "graph/mutators/bi_synthetic_data_type_mutator.h"

#include "graph/bi_graphBuilder.h"
#include "graph/bi_itensorAccessor.h"
#include "graph/bi_logger.h"
#include "graph/nodes/Nodes.h"
#include "graph/bi_utils.h"

#include "support/bi_cast.hpp"

#include <set>

namespace BatmanInfer {

namespace graph {

namespace {

    /** Empty accessor class */
    class EmptyAccessor final : public graph::BIITensorAccessor
    {
    public:
        /** Default Constructor */
        EmptyAccessor() = default;

        // Inherited methods overriden:
        bool access_tensor(BIITensor &tensor) override
        {
            BI_COMPUTE_UNUSED(tensor);
            return true;
        }
    };

    /** Check if the mutation pass can be applied
     *
     * @param[in] g Graph the mutation pass need to be applied on
     *
     * @return True if the pass can be applied else false
     */
    bool is_mutation_supported(BIGraph &g)
    {
        const std::set<BINodeType> unsupported_node_types =
            {BINodeType::DetectionOutputLayer, BINodeType::NormalizationLayer, BINodeType::PriorBoxLayer};

        for (const auto &utype : unsupported_node_types)
        {
            if (!g.nodes(utype).empty())
            {
                return false;
            }
        }
        return true;
    }

    /** Remove nodes that get optimized out during conversion
     *
     * @param[in, out] g Graph to remove the nodes from.
     */
    void remove_optimized_nodes(BIGraph &g)
    {
        const std::set<BINodeType> optimized_node_types = {BINodeType::BatchNormalizationLayer};

        for (const auto &opt_type : optimized_node_types)
        {
            const std::vector<NodeID> opt_nodes_ids = g.nodes(opt_type);
            for (const auto &node_id : opt_nodes_ids)
            {
                BIINode *node = g.node(node_id);

                // Get input edge
                BIEdge *input_edge = node->input_edge(0);
                BI_COMPUTE_ERROR_ON(input_edge == nullptr);

                // Get producer node
                BIINode     *producer         = input_edge->producer();
                const EdgeID producer_edge_id = input_edge->producer_idx();
                BI_COMPUTE_ERROR_ON(producer == nullptr);

                // Get driving nodes
                std::vector<BINodeIdxPair> driving_nodes = get_driving_nodes(*node);

                // Remove node
                g.remove_node(node->id());

                // Update connections
                for (auto &driving_node : driving_nodes)
                {
                    g.add_connection(producer->id(), producer_edge_id, driving_node.node_id, driving_node.index);
                }
            }
        }
    }

    /** Convert tensor meta-data
     *
     * @param[in,out] g Graph to convert tensors of.
     */
    void convert_tensors(BIGraph &g, BIDataType data_type)
    {
        auto &tensors = g.tensors();
        for (auto &tensor : tensors)
        {
            if (tensor != nullptr)
            {
                switch (data_type)
                {
                    case BIDataType::QASYMM8:
                    case BIDataType::QASYMM8_SIGNED:
                    {
                        tensor->desc().quant_info = BIQuantizationInfo(0.125f, -10);
                        break;
                    }
                    default:
                    {
                        BI_COMPUTE_ERROR("Unsupported mutation type");
                        break;
                    }
                }
                tensor->desc().data_type = data_type;
            }
        }
    }

    /** Convert special node
     *
     * @param[in,out] g                  Graph to convert tensors of.
     * @param[in]     f                  Conversion function.
     */
    template <typename NT>
    void convert_special_node(BIGraph &g, std::function<bool(BIINode *, BITensor *)> const &f)
    {
        const std::vector<NodeID> nodes_ids = g.nodes(NT::node_type);
        for (const auto &nodes_id : nodes_ids)
        {
            BIINode *node = BatmanInfer::utils::cast::polymorphic_downcast<NT *>(g.node(nodes_id));
            BI_COMPUTE_ERROR_ON(node == nullptr);

            BITensor *output_tensor = node->output(0);
            BI_COMPUTE_ERROR_ON(output_tensor == nullptr);

            f(node, output_tensor);
        }
    }

    /** Converts special tensors
     *
     * @param[in,out] g Graph to convert tensors of.
     */
    void convert_special_tensors(BIGraph &g)
    {
        auto softmax_func = [](BIINode *node, BITensor *tensor)
        {
            BI_COMPUTE_UNUSED(node);
            if (tensor->desc().data_type == BIDataType::QASYMM8)
            {
                tensor->desc().quant_info = BIQuantizationInfo(1.f / 256.f, 0);
            }
            else if (tensor->desc().data_type == BIDataType::QASYMM8_SIGNED)
            {
                tensor->desc().quant_info = BIQuantizationInfo(1.f / 256.f, -128);
            }
            return true;
        };

        auto act_func = [](BIINode *node, BITensor *tensor)
        {
            auto *act_node = BatmanInfer::utils::cast::polymorphic_downcast<ActivationLayerNode *>(node);
            if (tensor->desc().data_type == BIDataType::QASYMM8)
            {
                if (act_node->activation_info().activation() == BIActivationLayerInfo::ActivationFunction::TANH)
                {
                    tensor->desc().quant_info = BIQuantizationInfo(1.f / 128.f, 128);
                }
                else if (act_node->activation_info().activation() == BIActivationLayerInfo::ActivationFunction::LOGISTIC)
                {
                    tensor->desc().quant_info = BIQuantizationInfo(1.f / 256.f, 0);
                }
            }
            else if (tensor->desc().data_type == BIDataType::QASYMM8_SIGNED)
            {
                if (act_node->activation_info().activation() == BIActivationLayerInfo::ActivationFunction::TANH)
                {
                    tensor->desc().quant_info = BIQuantizationInfo(1.f / 128.f, 0);
                }
                else if (act_node->activation_info().activation() == BIActivationLayerInfo::ActivationFunction::LOGISTIC)
                {
                    tensor->desc().quant_info = BIQuantizationInfo(1.f / 256.f, -128);
                }
            }
            return true;
        };

        convert_special_node<ActivationLayerNode>(g, act_func);
        convert_special_node<SoftmaxLayerNode>(g, softmax_func);
    }

    /** Handle nodes with bias
     *
     * @note Special tensors are for now biases that the data type differ
     *
     * @param[in,out] g Graph to convert tensors of.
     */
    void handle_nodes_with_bias(BIGraph &g)
    {
        const std::set<BINodeType> special_node_types = {
            BINodeType::ConvolutionLayer, BINodeType::DeconvolutionLayer,
            BINodeType::DepthwiseConvolutionLayer, BINodeType::FullyConnectedLayer};

        for (const auto &spc_type : special_node_types)
        {
            const std::vector<NodeID> scp_nodes_ids = g.nodes(spc_type);
            for (const auto &node_id : scp_nodes_ids)
            {
                BIINode *node = g.node(node_id);
                if (node != nullptr)
                {
                    BITensor *tensor = node->input(2);
                    if (tensor != nullptr)
                    {
                        tensor->desc().data_type = BIDataType::S32;
                    }
                    else
                    {
                        auto params = node->common_node_params();
                        params.name = params.name.empty() ? "" : params.name + "Bias";

                        BITensorDescriptor b_desc= node->input(1)->desc();
                        auto depth =
                            b_desc.shape[get_dimension_idx(b_desc.layout, BIDataLayoutDimension::BATCHES)];
                        b_desc.shape = BITensorShape(depth);

                        auto accessor = std::make_unique<EmptyAccessor>();
                        auto b_nid    = GraphBuilder::add_const_node(g, params, b_desc, std::move(accessor));
                        g.add_connection(b_nid, 0, node_id, 2);
                    }
                }
            }
        }
    }

} // namespace

    BISyntheticDataTypeMutator::BISyntheticDataTypeMutator(BIDataType mutate_type) : _mutate_type{mutate_type}
    {
    }

    const char *BISyntheticDataTypeMutator::name()
    {
        return "BISyntheticDataTypeMutator";
    }

    BIIGraphMutator::MutationType BISyntheticDataTypeMutator::type() const
    {
        return BIIGraphMutator::MutationType::IR;
    }

    void BISyntheticDataTypeMutator::mutate(BIGraph &g)
    {
        if (is_mutation_supported(g))
        {
            // Remove nodes that get optimized out (e.g. BatchNorm)
            remove_optimized_nodes(g);

            // Convert tensor
            convert_tensors(g, _mutate_type);
            convert_special_tensors(g);

            // Handle special nodes
            handle_nodes_with_bias(g);
        }
        else
        {
            BI_COMPUTE_LOG_GRAPH_VERBOSE("Synthetic data type mutator couldn't be applied" << std::endl);
        }
    }

} // namespace graph

} // namespace BatmanInfer

