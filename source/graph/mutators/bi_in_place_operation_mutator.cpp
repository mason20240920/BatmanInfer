//
// Created by holynova on 25-3-12.
//

#include "graph/mutators/bi_in_place_operation_mutator.h"

#include "data/core/bi_helpers.hpp"
#include "data/core/bi_vlidate.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_logger.h"
#include "graph/nodes/DepthwiseConvolutionLayerNode.h"
#include "graph/nodes/FusedDepthwiseConvolutionBatchNormalizationNode.h"

#include "support/bi_cast.hpp"

using namespace BatmanInfer::utils::cast;

namespace BatmanInfer {

namespace graph {

namespace {

    // Check if the output edges of the parent node are separate tensors. If not,
    // it means the same output is connected to multiple nodes and computations on
    // these nodes cannot be done in-place.
    bool output_edges_are_separate_tensors(BIGraph &g, const BIEdge *input_edge)
    {
        const auto parent_node   = input_edge->producer();
        const auto input_tensor  = input_edge->tensor();
        const auto input_edge_id = input_edge->id();

        if (parent_node == nullptr)
        {
            return false;
        }

        const auto output_edges = parent_node->output_edges();

        // If the output is connected to only one edge, then computations can
        // be done in-place.
        if (output_edges.size() == 1)
        {
            return true;
        }

        return std::all_of(output_edges.begin(), output_edges.end(),
                           [&](const EdgeID &edge_id)
                           {
                               // Skip check on current input edge
                               if (edge_id == input_edge_id)
                               {
                                   return true;
                               }

                               auto edge = g.edge(edge_id);
                               return edge->tensor() != input_tensor;
                           });
    }

    // If do in-place calculation, then need to use the new output and inherit original output's accessor
    void set_new_output_and_inherit_accessor(std::unique_ptr<BIINode> &node, BITensor *orig_output, BITensor *new_output)
    {
        BI_COMPUTE_LOG_GRAPH_INFO("Switching to in-place computation for the node with ID : "
                                  << node->id() << " and name : " << node->name() << std::endl);
        // Update accessor
        new_output->set_accessor(orig_output->extract_accessor());
        // Update output
        node->set_output_tensor(new_output->id(), 0);
    }

    // Try to mutate the node to perform the depthwise in-place calculation
    void try_in_place_depthwiseconv(std::unique_ptr<BIINode> &node)
    {
        // Get input edge
        BIEdge *input_edge  = node->input_edge(0);
        BIEdge *weight_edge = node->input_edge(1);
        BI_COMPUTE_ERROR_ON(input_edge == nullptr || weight_edge == nullptr);

        auto input_tensor  = input_edge->tensor();
        auto weight_tensor = weight_edge->tensor();
        BI_COMPUTE_ERROR_ON(input_tensor == nullptr || weight_tensor == nullptr);

        const auto input_shape = input_tensor->desc().shape;
        const auto qinfo_input = input_tensor->desc().quant_info;

        const auto weight_shape  = weight_tensor->desc().shape;
        const auto weight_layout = weight_tensor->desc().layout;

        // Extract PadStrideInfo and depth multiplier
        BIPadStrideInfo conv_info{};
        unsigned int  depth_multiplier{};
        if (node->type() == BINodeType::FusedDepthwiseConvolutionBatchNormalizationLayer)
        {
            conv_info =
                polymorphic_downcast<FusedDepthwiseConvolutionBatchNormalizationNode *>(node.get())->convolution_info();
            depth_multiplier =
                polymorphic_downcast<FusedDepthwiseConvolutionBatchNormalizationNode *>(node.get())->depth_multiplier();
        }
        else if (node->type() == BINodeType::DepthwiseConvolutionLayer)
        {
            conv_info        = polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node.get())->convolution_info();
            depth_multiplier = polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node.get())->depth_multiplier();
        }

        // Get current output tensor
        auto current_output_tensor = node->output(0);
        BI_COMPUTE_ERROR_ON(current_output_tensor == nullptr);
        const auto out_shape = current_output_tensor->desc().shape;
        const auto qinfo_out = current_output_tensor->desc().quant_info;

        bool input_can_in_place = !BatmanInfer::detail::have_different_dimensions(out_shape, input_shape, 0) &&
                                  (qinfo_input == qinfo_out) && (input_tensor->accessor() == nullptr);

        // Specify conditions with which input can be in-placed
        input_can_in_place &= weight_layout == input_tensor->desc().layout && weight_layout == BIDataLayout::NHWC;

        const int  weights_width_idx  = get_data_layout_dimension_index(weight_layout, BIDataLayoutDimension::WIDTH);
        const int  weights_height_idx = get_data_layout_dimension_index(weight_layout, BIDataLayoutDimension::HEIGHT);
        const bool is_1x1             = weight_shape[weights_width_idx] == 1U && weight_shape[weights_height_idx] == 1U;
        input_can_in_place &= is_1x1;

        input_can_in_place &= depth_multiplier == 1;
        input_can_in_place &= conv_info.stride() == std::make_pair(1U, 1U);
        input_can_in_place &= !conv_info.has_padding();
        // NOTE: Dilation should also be (1, 1). However currently dilation is not supported in the depthwise conv node

        if (input_can_in_place)
        {
            set_new_output_and_inherit_accessor(node, current_output_tensor, input_tensor);
        }
        else
        {
            BI_COMPUTE_LOG_GRAPH_VERBOSE("Prevented in-place operation as there is an accessor bound to the input tensor "
                                         "or the quantization info are different.\n");
        }
    }

    // Try to mutate the node to perform the elementwise in-place calculation
    void try_in_place_elementwise(std::unique_ptr<BIINode> &node)
    {
        // Get input edge
        BIEdge *input0_edge = node->input_edge(0);
        BIEdge *input1_edge = node->input_edge(1);
        BI_COMPUTE_ERROR_ON(input0_edge == nullptr || input1_edge == nullptr);

        auto input0_tensor = input0_edge->tensor();
        auto input1_tensor = input1_edge->tensor();
        BI_COMPUTE_ERROR_ON(input0_tensor == nullptr || input1_tensor == nullptr);

        const auto shape0 = input0_tensor->desc().shape;
        const auto shape1 = input1_tensor->desc().shape;
        const auto qinfo0 = input0_tensor->desc().quant_info;
        const auto qinfo1 = input1_tensor->desc().quant_info;

        const BITensorShape out_shape = BITensorShape::broadcast_shape(shape0, shape1);
        // Inputs are not broadcast compatible
        if (out_shape.total_size() == 0)
        {
            return;
        }

        // Get current output tensor
        auto current_output_tensor = node->output(0);
        BI_COMPUTE_ERROR_ON(current_output_tensor == nullptr);
        const auto qinfo_out = current_output_tensor->desc().quant_info;

        // Can do in place, if the input has same shape as output, has same quntisation info as output, has same data type as output and input doesn't have accessor.
        bool input0_can_in_place = !BatmanInfer::detail::have_different_dimensions(out_shape, shape0, 0) &&
                                   (qinfo0 == qinfo_out) &&
                                   (input0_tensor->desc().data_type == current_output_tensor->desc().data_type) &&
                                   (input0_tensor->accessor() == nullptr);
        bool input1_can_in_place = !BatmanInfer::detail::have_different_dimensions(out_shape, shape1, 0) &&
                                   (qinfo1 == qinfo_out) &&
                                   (input1_tensor->desc().data_type == current_output_tensor->desc().data_type) &&
                                   (input1_tensor->accessor() == nullptr);

        if (input0_can_in_place)
        {
            set_new_output_and_inherit_accessor(node, current_output_tensor, input0_tensor);
        }
        else if (input1_can_in_place)
        {
            set_new_output_and_inherit_accessor(node, current_output_tensor, input1_tensor);
        }
        else
        {
            BI_COMPUTE_LOG_GRAPH_VERBOSE("Prevented in-place operation as there is an accessor bound to the input tensor "
                                         "or the quantization info are different.\n");
        }
    }

} // namespace

    const char *BIInPlaceOperationMutator::name()
    {
        return "BIInPlaceOperationMutator";
    }

    BIIGraphMutator::MutationType BIInPlaceOperationMutator::type() const
    {
        return BIIGraphMutator::MutationType::Backend;
    }

    void BIInPlaceOperationMutator::mutate(BIGraph &g)
    {
        std::set<BINodeType> in_place_nodes = {BINodeType::ActivationLayer,
                                               BINodeType::BatchNormalizationLayer,
                                               BINodeType::EltwiseLayer,
                                               BINodeType::UnaryEltwiseLayer,
                                               BINodeType::DepthwiseConvolutionLayer,
                                               BINodeType::FusedDepthwiseConvolutionBatchNormalizationLayer,
                                               BINodeType::PrintLayer};

        // Not interested in the order of nodes
        for (auto &node : g.nodes())
        {
            if (node && in_place_nodes.find(node->type()) != std::end(in_place_nodes))
            {
                // Get input edge
                BIEdge *input_edge = node->input_edge(0);

                // Check if parent has a single output if yes then force in place calculation else not
                if ((input_edge != nullptr) && output_edges_are_separate_tensors(g, input_edge))
                {
                    if (node->type() == BINodeType::EltwiseLayer)
                    {
                        try_in_place_elementwise(node);
                    }
                    else if (node->type() == BINodeType::FusedDepthwiseConvolutionBatchNormalizationLayer ||
                             node->type() == BINodeType::DepthwiseConvolutionLayer)
                    {
                        try_in_place_depthwiseconv(node);
                    }
                    else
                    {
                        // Get current and new output tensors
                        auto current_output_tensor = node->output(0);
                        auto new_output_tensor     = input_edge->tensor();

                        BI_COMPUTE_ERROR_ON(current_output_tensor == nullptr || new_output_tensor == nullptr);

                        // Prevent in-place operation if there is an accessor bound to the in-place tensor or quantization info are different
                        if (new_output_tensor->accessor() != nullptr ||
                            current_output_tensor->desc().quant_info != new_output_tensor->desc().quant_info)
                        {
                            BI_COMPUTE_LOG_GRAPH_VERBOSE("Prevented in-place operation as there is an accessor bound to "
                                                         "the input tensor or the quantization info are different.\n");
                        }
                        else
                        {
                            set_new_output_and_inherit_accessor(node, current_output_tensor, new_output_tensor);
                        }
                    }
                }
            }
        }
    }

} // namespace graph

} // namespace BatmanInfer
