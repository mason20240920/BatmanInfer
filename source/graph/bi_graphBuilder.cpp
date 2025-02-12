//
// Created by holynova on 2025/2/7.
//

#include "graph/bi_graphBuilder.h"

#include "data/core/utils/data_type_utils.hpp"
#include "graph/algorithms/bi_algorithms.h"
#include "graph/bi_graph.h"
#include "graph/nodes/Nodes.h"
#include "graph/bi_utils.h"
#include "support/bi_toolchain_support.hpp"

namespace BatmanInfer {

namespace graph {

namespace {

    inline void check_nodeidx_pair(const BINodeIdxPair &pair, const BIGraph &g)
    {
        BI_COMPUTE_UNUSED(pair);
        BI_COMPUTE_UNUSED(g);
        BI_COMPUTE_ERROR_ON((pair.node_id >= g.nodes().size()) || (g.node((pair).node_id) == nullptr) ||
                             (pair.index >= g.node(pair.node_id)->num_outputs()));
    }

    BIStatus set_node_params(BIGraph &g, NodeID nid, BINodeParams &params)
    {
        BIINode *node = g.node(nid);
        BI_COMPUTE_RETURN_ERROR_ON(!node);

        node->set_common_node_parameters(params);

        return BIStatus{};
    }

    BIStatus set_accessor_on_node(BIGraph &g, NodeID nid, bool is_output, size_t idx, BIITensorAccessorUPtr accessor)
    {
        BIINode *node = g.node(nid);
        BI_COMPUTE_RETURN_ERROR_ON(!node);

        BITensor *tensor = is_output ? node->output(idx) : node->input(idx);
        BI_COMPUTE_RETURN_ERROR_ON(!tensor);

        tensor->set_accessor(std::move(accessor));

        return BIStatus{};
    }

    NodeID add_const_node_with_name(BIGraph &g, BINodeParams params, const std::string &name,
                                    const BITensorDescriptor &desc, BIITensorAccessorUPtr accessor)
    {
        params.name = params.name.empty() ? "" : params.name + name;
        auto nid    = GraphBuilder::add_const_node(g, params, desc, std::move(accessor));
        set_node_params(g, nid, params);
        return nid;
    }

    template <typename NT, typename... Args>
    NodeID create_simple_single_input_output_node(BIGraph &g, BINodeParams &params, BINodeIdxPair input, Args &&...args)
    {
        check_nodeidx_pair(input, g);

        NodeID nid = g.add_node<NT>(std::forward<Args>(args)...);
        g.add_connection(input.node_id, input.index, nid, 0);
        set_node_params(g, nid, params);

        return nid;
    }

    template <typename NT, typename... Args>
    NodeID create_simple_multiple_input_single_output_node(BIGraph                          &g,
                                                           BINodeParams                     &params,
                                                           const std::vector<BINodeIdxPair> &inputs,
                                                           Args &&...args)
    {
        BI_COMPUTE_ERROR_ON(inputs.size() == 0);

        NodeID nid = g.add_node<NT>(std::forward<Args>(args)...);

        unsigned int i = 0;
        for (const auto &input : inputs)
        {
            check_nodeidx_pair(input, g);
            g.add_connection(input.node_id, input.index, nid, i++);
        }
        set_node_params(g, nid, params);

        return nid;
    }

} // namespace unnamed

    NodeID GraphBuilder::add_const_node(BIGraph &g, BINodeParams params, const BITensorDescriptor &desc,
                                        BIITensorAccessorUPtr accessor)
    {
        auto nid = g.add_node<ConstNode>(desc);
        set_node_params(g, nid, params);
        set_accessor_on_node(g, nid, true, 0, std::move(accessor));
        return nid;
    }

    NodeID GraphBuilder::add_input_node(BIGraph &g, BINodeParams params, const BITensorDescriptor &desc,
                                        BIITensorAccessorUPtr accessor)
    {
        auto nid = g.add_node<InputNode>(desc);
        set_node_params(g, nid, params);
        set_accessor_on_node(g, nid, true, 0, std::move(accessor));
        return nid;
    }

    NodeID GraphBuilder::add_output_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                         BIITensorAccessorUPtr accessor)
    {
        check_nodeidx_pair(input, g);

        NodeID nid = g.add_node<OutputNode>();
        g.add_connection(input.node_id, input.index, nid, 0);
        set_node_params(g, nid, params);
        set_accessor_on_node(g, nid, false, 0, std::move(accessor));

        return nid;
    }

    NodeID GraphBuilder::add_activation_node(BIGraph                  &g,
                                             BINodeParams              params,
                                             BINodeIdxPair             input,
                                             BIActivationLayerInfo     act_info,
                                             const BIQuantizationInfo &out_quant_info)
    {
        return create_simple_single_input_output_node<ActivationLayerNode>(g, params, input, act_info, out_quant_info);
    }

    NodeID GraphBuilder::add_arg_min_max_node(BIGraph                  &g,
                                              BINodeParams              params,
                                              BINodeIdxPair             input,
                                              BIReductionOperation      op,
                                              unsigned int              axis,
                                              BIDataType                out_data_type,
                                              const BIQuantizationInfo &out_quant_info)
    {
        return create_simple_single_input_output_node<ArgMinMaxLayerNode>(g, params, input, op, axis, out_data_type,
                                                                          out_quant_info);
    }

    NodeID GraphBuilder::add_batch_normalization_node(BIGraph              &g,
                                                      BINodeParams          params,
                                                      BINodeIdxPair         input,
                                                      float                 epsilon,
                                                      BIITensorAccessorUPtr mean_accessor,
                                                      BIITensorAccessorUPtr var_accessor,
                                                      BIITensorAccessorUPtr beta_accessor,
                                                      BIITensorAccessorUPtr gamma_accessor)
    {
        check_nodeidx_pair(input, g);

        bool has_beta  = (beta_accessor != nullptr);
        bool has_gamma = (gamma_accessor != nullptr);

        // Get input tensor descriptor
        const BITensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

        // Calculate Common Descriptor
        BITensorDescriptor common_desc = input_tensor_desc;
        common_desc.shape              = BITensorShape(get_dimension_size(input_tensor_desc,
                                                             BIDataLayoutDimension::CHANNEL));

        // Create mean and var nodes
        auto mean_nid = add_const_node_with_name(g, params, "Mean", common_desc, std::move(mean_accessor));
        auto var_nid  = add_const_node_with_name(g, params, "Variance", common_desc, std::move(var_accessor));

        // Create beta node
        NodeID beta_nid = EmptyNodeID;
        if (has_beta)
        {
            beta_nid = add_const_node_with_name(g, params, "Beta", common_desc, std::move(beta_accessor));
        }

        // Create gamma node
        NodeID gamma_nid = EmptyNodeID;
        if (has_gamma)
        {
            gamma_nid = add_const_node_with_name(g, params, "Gamma", common_desc, std::move(gamma_accessor));
        }

        // Create batch normalization node and add connections
        NodeID batch_norm_nid = g.add_node<BatchNormalizationLayerNode>(epsilon);
        g.add_connection(input.node_id, input.index, batch_norm_nid, 0);
        g.add_connection(mean_nid, 0, batch_norm_nid, 1);
        g.add_connection(var_nid, 0, batch_norm_nid, 2);
        if (has_beta)
        {
            g.add_connection(beta_nid, 0, batch_norm_nid, 3);
        }
        if (has_gamma)
        {
            g.add_connection(gamma_nid, 0, batch_norm_nid, 4);
        }
        set_node_params(g, batch_norm_nid, params);

        return batch_norm_nid;
    }

    NodeID GraphBuilder::add_bounding_box_transform_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                                         BINodeIdxPair deltas, BIBoundingBoxTransformInfo info)
    {
        check_nodeidx_pair(input, g);
        check_nodeidx_pair(deltas, g);

        NodeID nid = g.add_node<BoundingBoxTransformLayerNode>(info);

        g.add_connection(input.node_id, input.index, nid, 0);
        g.add_connection(deltas.node_id, deltas.index, nid, 1);

        set_node_params(g, nid, params);
        return nid;
    }

    NodeID GraphBuilder::add_channel_shuffle_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                                  unsigned int num_groups)
    {
        return create_simple_single_input_output_node<ChannelShuffleLayerNode>(g, params, input, num_groups);
    }

    NodeID GraphBuilder::add_convolution_node(BIGraph                  &g,
                                              BINodeParams              params,
                                              BINodeIdxPair             input,
                                              Size2D                    kernel_spatial_extend,
                                              unsigned int              depth,
                                              BIPadStrideInfo           conv_info,
                                              unsigned int              num_groups,
                                              BIConvolutionMethod       method,
                                              BIFastMathHint            fast_math_hint,
                                              BIITensorAccessorUPtr     weights_accessor,
                                              BIITensorAccessorUPtr     bias_accessor,
                                              const BIQuantizationInfo &weights_quant_info,
                                              const BIQuantizationInfo &out_quant_info)
    {
        check_nodeidx_pair(input, g);
        BI_COMPUTE_ERROR_ON(depth == 0);
        BI_COMPUTE_ERROR_ON((kernel_spatial_extend.width == 0) || (kernel_spatial_extend.height == 0));

        bool has_bias = (bias_accessor != nullptr);

        // Get input tensor descriptor
        const BITensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
        const BIDataLayout       input_data_layout = input_tensor_desc.layout;

        // Create weights node
        BITensorDescriptor w_desc = input_tensor_desc;
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::WIDTH),
                         kernel_spatial_extend.width);
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::HEIGHT),
                         kernel_spatial_extend.height);
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::CHANNEL),
                         get_dimension_size(input_tensor_desc, BIDataLayoutDimension::CHANNEL) / num_groups);
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::BATCHES), depth);
        if (!weights_quant_info.empty())
        {
            w_desc.quant_info = weights_quant_info;
        }

        NodeID w_nid = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

        // Create bias nodes
        NodeID b_nid = EmptyNodeID;
        if (has_bias)
        {
            BITensorDescriptor b_desc = input_tensor_desc;
            b_desc.shape              = BITensorShape(depth);
            if (is_data_type_quantized_asymmetric(input_tensor_desc.data_type))
            {
                b_desc.data_type = BIDataType::S32;
            }
            b_nid = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
        }

        // Create convolution node and connect
        NodeID conv_nid = g.add_node<ConvolutionLayerNode>(conv_info, num_groups, method, fast_math_hint, out_quant_info);
        g.add_connection(input.node_id, input.index, conv_nid, 0);
        g.add_connection(w_nid, 0, conv_nid, 1);
        if (has_bias)
        {
            g.add_connection(b_nid, 0, conv_nid, 2);
        }
        set_node_params(g, conv_nid, params);

        return conv_nid;
    }

    NodeID GraphBuilder::add_deconvolution_node(BIGraph              &g,
                                                BINodeParams          params,
                                                BINodeIdxPair         input,
                                                Size2D                kernel_spatial_extend,
                                                unsigned int          depth,
                                                BIPadStrideInfo       deconv_info,
                                                BIITensorAccessorUPtr weights_accessor,
                                                BIITensorAccessorUPtr bias_accessor)
    {
        check_nodeidx_pair(input, g);
        BI_COMPUTE_ERROR_ON(depth == 0);
        BI_COMPUTE_ERROR_ON((kernel_spatial_extend.width == 0) || (kernel_spatial_extend.height == 0));

        bool has_bias = (bias_accessor != nullptr);

        // Get input tensor descriptor
        const BITensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
        const BIDataLayout       input_data_layout = input_tensor_desc.layout;

        // Create weights node
        BITensorDescriptor w_desc = input_tensor_desc;
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::WIDTH),
                         kernel_spatial_extend.width);
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::HEIGHT),
                         kernel_spatial_extend.height);
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::CHANNEL),
                         get_dimension_size(input_tensor_desc, BIDataLayoutDimension::CHANNEL));
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::BATCHES), depth);

        NodeID w_nid = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

        // Create bias nodes
        NodeID b_nid = EmptyNodeID;
        if (has_bias)
        {
            BITensorDescriptor b_desc = input_tensor_desc;
            b_desc.shape              = BITensorShape(depth);
            if (is_data_type_quantized_asymmetric(input_tensor_desc.data_type))
            {
                b_desc.data_type = BIDataType::S32;
            }
            b_nid = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
        }

        // Create convolution node and connect
        NodeID deconv_nid = g.add_node<DeconvolutionLayerNode>(descriptors::DeconvolutionLayerDescriptor{deconv_info});
        g.add_connection(input.node_id, input.index, deconv_nid, 0);
        g.add_connection(w_nid, 0, deconv_nid, 1);
        if (has_bias)
        {
            g.add_connection(b_nid, 0, deconv_nid, 2);
        }
        set_node_params(g, deconv_nid, params);

        return deconv_nid;
    }

    NodeID GraphBuilder::add_concatenate_node(BIGraph                                  &g,
                                              BINodeParams                              params,
                                              const std::vector<BINodeIdxPair>         &inputs,
                                              const descriptors::ConcatLayerDescriptor &concat_descriptor)
    {
        return create_simple_multiple_input_single_output_node<ConcatenateLayerNode>(g, params, inputs, inputs.size(),
                                                                                     concat_descriptor);
    }

    NodeID GraphBuilder::add_depthwise_convolution_node(BIGraph                     &g,
                                                        BINodeParams                 params,
                                                        BINodeIdxPair                input,
                                                        Size2D                       kernel_spatial_extend,
                                                        BIPadStrideInfo              conv_info,
                                                        int                          depth_multiplier,
                                                        BIDepthwiseConvolutionMethod method,
                                                        BIITensorAccessorUPtr        weights_accessor,
                                                        BIITensorAccessorUPtr        bias_accessor,
                                                        const BIQuantizationInfo    &quant_info,
                                                        const BIQuantizationInfo    &out_quant_info)
    {
        check_nodeidx_pair(input, g);
        BI_COMPUTE_ERROR_ON((kernel_spatial_extend.width == 0) || (kernel_spatial_extend.height == 0));

        bool has_bias = (bias_accessor != nullptr);

        // Get input tensor descriptor
        const BITensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
        const BIDataLayout       input_data_layout = input_tensor_desc.layout;

        // Create weights node
        BITensorDescriptor w_desc = input_tensor_desc;
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::WIDTH),
                         kernel_spatial_extend.width);
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::HEIGHT),
                         kernel_spatial_extend.height);
        w_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::CHANNEL),
                         get_dimension_size(input_tensor_desc, BIDataLayoutDimension::CHANNEL) * depth_multiplier);
        if (!quant_info.empty())
        {
            w_desc.quant_info = quant_info;
        }

        NodeID w_nid = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

        // Create bias nodes
        NodeID b_nid = EmptyNodeID;
        if (has_bias)
        {
            BITensorDescriptor b_desc = input_tensor_desc;
            b_desc.shape = BITensorShape(
                get_dimension_size(input_tensor_desc, BIDataLayoutDimension::CHANNEL) * depth_multiplier);

            if (is_data_type_quantized_asymmetric(b_desc.data_type))
            {
                b_desc.data_type = BIDataType::S32;
            }

            b_nid = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
        }

        // Create convolution node and connect
        NodeID conv_nid = g.add_node<DepthwiseConvolutionLayerNode>(conv_info, depth_multiplier, method, out_quant_info);
        g.add_connection(input.node_id, input.index, conv_nid, 0);
        g.add_connection(w_nid, 0, conv_nid, 1);
        if (has_bias)
        {
            g.add_connection(b_nid, 0, conv_nid, 2);
        }
        set_node_params(g, conv_nid, params);

        return conv_nid;
    }

    NodeID GraphBuilder::add_depth_to_space_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                                 int32_t block_shape)
    {
        return create_simple_single_input_output_node<DepthToSpaceLayerNode>(g, params, input, block_shape);
    }

    NodeID GraphBuilder::add_dequantization_node(BIGraph &g, BINodeParams params, BINodeIdxPair input)
    {
        return create_simple_single_input_output_node<DequantizationLayerNode>(g, params, input);
    }

    NodeID GraphBuilder::add_detection_output_node(BIGraph                          &g,
                                                   BINodeParams                      params,
                                                   BINodeIdxPair                     input_loc,
                                                   BINodeIdxPair                     input_conf,
                                                   BINodeIdxPair                     input_priorbox,
                                                   const BIDetectionOutputLayerInfo &detect_info)
    {
        check_nodeidx_pair(input_loc, g);
        check_nodeidx_pair(input_conf, g);
        check_nodeidx_pair(input_priorbox, g);

        // Create detection_output node and connect
        NodeID detect_nid = g.add_node<DetectionOutputLayerNode>(detect_info);
        g.add_connection(input_loc.node_id, input_loc.index, detect_nid, 0);
        g.add_connection(input_conf.node_id, input_conf.index, detect_nid, 1);
        g.add_connection(input_priorbox.node_id, input_priorbox.index, detect_nid, 2);

        set_node_params(g, detect_nid, params);

        return detect_nid;
    }

    NodeID GraphBuilder::add_detection_post_process_node(BIGraph                               &g,
                                                         BINodeParams                           params,
                                                         BINodeIdxPair                          input_box_encoding,
                                                         BINodeIdxPair                          input_class_prediction,
                                                         const BIDetectionPostProcessLayerInfo &detect_info,
                                                         BIITensorAccessorUPtr                  anchors_accessor,
                                                         const BIQuantizationInfo              &anchor_quant_info)
    {
        check_nodeidx_pair(input_box_encoding, g);
        check_nodeidx_pair(input_class_prediction, g);

        // Get input tensor descriptor
        const BITensorDescriptor input_box_encoding_tensor_desc =
            get_tensor_descriptor(g, g.node(input_box_encoding.node_id)->outputs()[0]);

        // Calculate anchor descriptor
        BITensorDescriptor anchor_desc = input_box_encoding_tensor_desc;
        if (!anchor_quant_info.empty())
        {
            anchor_desc.quant_info = anchor_quant_info;
        }

        // Create anchors nodes
        auto anchors_nid = add_const_node_with_name(g, params, "Anchors", anchor_desc, std::move(anchors_accessor));

        // Create detection_output node and connect
        NodeID detect_nid = g.add_node<DetectionPostProcessLayerNode>(detect_info);
        g.add_connection(input_box_encoding.node_id, input_box_encoding.index, detect_nid, 0);
        g.add_connection(input_class_prediction.node_id, input_class_prediction.index, detect_nid, 1);
        g.add_connection(anchors_nid, 0, detect_nid, 2);

        set_node_params(g, detect_nid, params);

        return detect_nid;
    }

    NodeID GraphBuilder::add_dummy_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BITensorShape shape)
    {
        return create_simple_single_input_output_node<DummyNode>(g, params, input, shape);
    }

    NodeID GraphBuilder::add_elementwise_node(BIGraph &g, BINodeParams params, BINodeIdxPair input0,
                                              BINodeIdxPair input1, BIEltwiseOperation operation)
    {
        check_nodeidx_pair(input0, g);
        check_nodeidx_pair(input1, g);

        NodeID nid = g.add_node<EltwiseLayerNode>(descriptors::EltwiseLayerDescriptor{operation});

        g.add_connection(input0.node_id, input0.index, nid, 0);
        g.add_connection(input1.node_id, input1.index, nid, 1);

        set_node_params(g, nid, params);

        return nid;
    }

    NodeID GraphBuilder::add_flatten_node(BIGraph &g, BINodeParams params, BINodeIdxPair input)
    {
        return create_simple_single_input_output_node<FlattenLayerNode>(g, params, input);
    }

    NodeID GraphBuilder::add_fully_connected_layer(BIGraph                        &g,
                                                   BINodeParams                    params,
                                                   BINodeIdxPair                   input,
                                                   unsigned int                    num_outputs,
                                                   NodeID                          weights_nid,
                                                   NodeID                          bias_nid,
                                                   const BIFullyConnectedLayerInfo fc_info,
                                                   const BIQuantizationInfo       &out_quant_info,
                                                   BIFastMathHint                  fast_math_hint)
    {
        check_nodeidx_pair(input, g);
        BI_COMPUTE_ERROR_ON(num_outputs == 0);
        BI_COMPUTE_ERROR_ON(weights_nid == EmptyNodeID);

        const bool has_bias = (bias_nid != EmptyNodeID);

        // Get input tensor descriptor
        const BITensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

        // Create fully connected node and connect
        NodeID fc_nid = g.add_node<FullyConnectedLayerNode>(num_outputs, out_quant_info, fc_info, fast_math_hint);
        g.add_connection(input.node_id, input.index, fc_nid, 0);
        g.add_connection(weights_nid, 0, fc_nid, 1);
        if (has_bias)
        {
            g.add_connection(bias_nid, 0, fc_nid, 2);
        }

        set_node_params(g, fc_nid, params);

        return fc_nid;
    }

    NodeID GraphBuilder::add_fully_connected_layer(BIGraph                        &g,
                                                   BINodeParams                    params,
                                                   BINodeIdxPair                   input,
                                                   unsigned int                    num_outputs,
                                                   BIITensorAccessorUPtr           weights_accessor,
                                                   BIITensorAccessorUPtr           bias_accessor,
                                                   const BIFullyConnectedLayerInfo fc_info,
                                                   const BIQuantizationInfo       &weights_quant_info,
                                                   const BIQuantizationInfo       &out_quant_info,
                                                   BIFastMathHint                  fast_math_hint)
    {
        check_nodeidx_pair(input, g);
        BI_COMPUTE_ERROR_ON(num_outputs == 0);

        bool has_bias = (bias_accessor != nullptr);

        // Get input tensor descriptor
        const BITensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

        // Create weights node
        BITensorDescriptor w_desc = FullyConnectedLayerNode::compute_weights_descriptor(input_tensor_desc, num_outputs,
                                                                                  fc_info, weights_quant_info);
        NodeID w_nid = add_const_node_with_name(g, params, "Weights", w_desc, std::move(weights_accessor));

        // Create bias nodes
        NodeID b_nid = EmptyNodeID;
        if (has_bias)
        {
            BITensorDescriptor b_desc = input_tensor_desc;
            b_desc.shape              = BITensorShape(num_outputs);
            if (is_data_type_quantized_asymmetric(input_tensor_desc.data_type))
            {
                b_desc.data_type = BIDataType::S32;
            }
            b_nid = add_const_node_with_name(g, params, "Bias", b_desc, std::move(bias_accessor));
        }

        // Create fully connected node and connect
        NodeID fc_nid = g.add_node<FullyConnectedLayerNode>(num_outputs, out_quant_info, fc_info, fast_math_hint);
        g.add_connection(input.node_id, input.index, fc_nid, 0);
        g.add_connection(w_nid, 0, fc_nid, 1);
        if (has_bias)
        {
            g.add_connection(b_nid, 0, fc_nid, 2);
        }

        set_node_params(g, fc_nid, params);

        return fc_nid;
    }

    NodeID GraphBuilder::add_generate_proposals_node(BIGraph                &g,
                                                     BINodeParams            params,
                                                     BINodeIdxPair           scores,
                                                     BINodeIdxPair           deltas,
                                                     BINodeIdxPair           anchors,
                                                     BIGenerateProposalsInfo info)
    {
        check_nodeidx_pair(scores, g);
        check_nodeidx_pair(deltas, g);
        check_nodeidx_pair(anchors, g);

        NodeID nid = g.add_node<GenerateProposalsLayerNode>(info);

        g.add_connection(scores.node_id, scores.index, nid, 0);
        g.add_connection(deltas.node_id, deltas.index, nid, 1);
        g.add_connection(anchors.node_id, anchors.index, nid, 2);

        set_node_params(g, nid, params);
        return nid;
    }

    NodeID GraphBuilder::add_l2_normalize_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, int axis,
                                               float epsilon)
    {
        return create_simple_single_input_output_node<L2NormalizeLayerNode>(g, params, input, axis, epsilon);
    }

    NodeID GraphBuilder::add_normalization_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                                BINormalizationLayerInfo norm_info)
    {
        return create_simple_single_input_output_node<NormalizationLayerNode>(g, params, input, norm_info);
    }

    NodeID GraphBuilder::add_normalize_planar_yuv_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                                       BIITensorAccessorUPtr mean_accessor,
                                                       BIITensorAccessorUPtr std_accessor)
    {
        check_nodeidx_pair(input, g);

        // Get input tensor descriptor
        const BITensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);

        // Calculate Common Descriptor
        BITensorDescriptor common_desc = input_tensor_desc;
        common_desc.shape = BITensorShape(get_dimension_size(input_tensor_desc, BIDataLayoutDimension::CHANNEL));

        // Create mean and std nodes
        auto mean_nid = add_const_node_with_name(g, params, "Mean", common_desc, std::move(mean_accessor));
        auto std_nid  = add_const_node_with_name(g, params, "Std", common_desc, std::move(std_accessor));

        // Create normalize planar YUV node and add connections
        NodeID norm_planar_yuv_nid = g.add_node<NormalizePlanarYUVLayerNode>();
        g.add_connection(input.node_id, input.index, norm_planar_yuv_nid, 0);
        g.add_connection(mean_nid, 0, norm_planar_yuv_nid, 1);
        g.add_connection(std_nid, 0, norm_planar_yuv_nid, 2);
        set_node_params(g, norm_planar_yuv_nid, params);

        return norm_planar_yuv_nid;
    }

    NodeID GraphBuilder::add_pad_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                      const PaddingList &paddings, BIPixelValue pad_value)
    {
        return create_simple_single_input_output_node<PadLayerNode>(g, params, input, paddings, pad_value);
    }

    NodeID GraphBuilder::add_permute_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                          PermutationVector perm, BIDataLayout layout)
    {
        return create_simple_single_input_output_node<PermuteLayerNode>(g, params, input, perm, layout);
    }

    NodeID GraphBuilder::add_prelu_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BINodeIdxPair alpha)
    {
        check_nodeidx_pair(input, g);
        check_nodeidx_pair(alpha, g);

        NodeID prelu_nid = g.add_node<PReluLayerNode>();
        g.add_connection(input.node_id, input.index, prelu_nid, 0);
        g.add_connection(alpha.node_id, alpha.index, prelu_nid, 1);

        set_node_params(g, prelu_nid, params);

        return prelu_nid;
    }

    NodeID GraphBuilder::add_pooling_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                          BIPoolingLayerInfo pool_info)
    {
        return create_simple_single_input_output_node<PoolingLayerNode>(g, params, input, pool_info);
    }

    NodeID GraphBuilder::add_print_node(BIGraph                                      &g,
                                        BINodeParams                                  params,
                                        BINodeIdxPair                                 input,
                                        std::ostream                                 &stream,
                                        const BIIOFormatInfo                         &format_info,
                                        const std::function<BIITensor *(BIITensor *)> transform)
    {
        return create_simple_single_input_output_node<PrintLayerNode>(g, params, input, stream, format_info, transform);
    }

    NodeID GraphBuilder::add_priorbox_node(BIGraph &g, BINodeParams params, BINodeIdxPair input0,
                                           BINodeIdxPair input1, const BIPriorBoxLayerInfo &prior_info)
    {
        check_nodeidx_pair(input0, g);
        check_nodeidx_pair(input1, g);

        // Create priorbox node and connect
        NodeID prior_nid = g.add_node<PriorBoxLayerNode>(prior_info);
        g.add_connection(input0.node_id, input0.index, prior_nid, 0);
        g.add_connection(input1.node_id, input1.index, prior_nid, 1);

        set_node_params(g, prior_nid, params);

        return prior_nid;
    }

    NodeID GraphBuilder::add_quantization_node(BIGraph                  &g,
                                               BINodeParams              params,
                                               BINodeIdxPair             input,
                                               const BIQuantizationInfo &out_quant_info)
    {
        return create_simple_single_input_output_node<QuantizationLayerNode>(g, params, input, out_quant_info);
    }

    NodeID GraphBuilder::add_reduction_operation_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                                      BIReductionOperation op, int axis, bool keep_dims)
    {
        return create_simple_single_input_output_node<ReductionLayerNode>(g, params, input, op, axis, keep_dims);
    }

    NodeID GraphBuilder::add_reorg_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, int stride)
    {
        return create_simple_single_input_output_node<ReorgLayerNode>(g, params, input, stride);
    }

    NodeID GraphBuilder::add_reshape_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BITensorShape shape)
    {
        return create_simple_single_input_output_node<ReshapeLayerNode>(g, params, input, shape);
    }

    NodeID GraphBuilder::add_resize_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                         BIInterpolationPolicy policy, float width_scale, float height_scale)
    {
        return create_simple_single_input_output_node<ResizeLayerNode>(g, params, input, policy, width_scale, height_scale);
    }

    NodeID GraphBuilder::add_roi_align_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                            BINodeIdxPair rois, BIROIPoolingLayerInfo pool_info)
    {
        check_nodeidx_pair(input, g);
        check_nodeidx_pair(rois, g);

        NodeID nid = g.add_node<ROIAlignLayerNode>(pool_info);

        g.add_connection(input.node_id, input.index, nid, 0);
        g.add_connection(rois.node_id, rois.index, nid, 1);

        set_node_params(g, nid, params);
        return nid;
    }

    NodeID GraphBuilder::add_scale_layer(BIGraph              &g,
                                         const BINodeParams   &params,
                                         BINodeIdxPair         input,
                                         BIITensorAccessorUPtr mul_accessor,
                                         BIITensorAccessorUPtr add_accessor)
    {
        check_nodeidx_pair(input, g);

        // Get input tensor descriptor
        const BITensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
        const BIDataLayout       input_data_layout = input_tensor_desc.layout;

        // Create mul node
        BITensorDescriptor mul_desc = input_tensor_desc;
        const size_t C = input_tensor_desc.shape[get_dimension_idx(input_data_layout, BIDataLayoutDimension::CHANNEL)];
        mul_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::WIDTH), 1);
        mul_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::HEIGHT), 1);
        mul_desc.shape.set(get_dimension_idx(input_data_layout, BIDataLayoutDimension::CHANNEL), C);
        NodeID      mul_const_nid   = add_const_node_with_name(g, params, "Mul", mul_desc, std::move(mul_accessor));
        BINodeIdxPair mul_const_nidxp = {mul_const_nid, 0};

        // Create add node
        BITensorDescriptor add_desc = mul_desc;
        NodeID add_const_nid   = add_const_node_with_name(g, params, "Add", add_desc, std::move(add_accessor));
        BINodeIdxPair add_const_nidxp = {add_const_nid, 0};

        // Create node and connect
        NodeID mul_node = GraphBuilder::add_elementwise_node(g, params, input, mul_const_nidxp, BIEltwiseOperation::Mul);
        BINodeIdxPair mulnode_nidxp = {mul_node, 0};
        NodeID add_node =
            GraphBuilder::add_elementwise_node(g, params, mulnode_nidxp, add_const_nidxp, BIEltwiseOperation::Add);

        return add_node;
    }

    NodeID GraphBuilder::add_softmax_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, float beta)
    {
        return create_simple_single_input_output_node<SoftmaxLayerNode>(g, params, input, beta);
    }

    NodeID GraphBuilder::add_slice_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                        BICoordinates &starts, BICoordinates &ends)
    {
        return create_simple_single_input_output_node<SliceLayerNode>(g, params, input, starts, ends);
    }

    NodeID GraphBuilder::add_split_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                        unsigned int num_splits, unsigned int axis)
    {
        return create_simple_single_input_output_node<SplitLayerNode>(g, params, input, num_splits, axis);
    }

    NodeID GraphBuilder::add_strided_slice_node(BIGraph                &g,
                                                BINodeParams            params,
                                                BINodeIdxPair           input,
                                                BICoordinates          &starts,
                                                BICoordinates          &ends,
                                                BiStrides              &strides,
                                                BIStridedSliceLayerInfo info)
    {
        return create_simple_single_input_output_node<StridedSliceLayerNode>(g, params, input, starts, ends,
                                                                           strides, info);
    }

    NodeID GraphBuilder::add_stack_node(BIGraph &g, BINodeParams params,
                                        const std::vector<BINodeIdxPair> &inputs, int axis)
    {
        return create_simple_multiple_input_single_output_node<StackLayerNode>(g, params, inputs, inputs.size(), axis);
    }

    NodeID GraphBuilder::add_yolo_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                       BIActivationLayerInfo act_info)
    {
        check_nodeidx_pair(input, g);

        // Get input tensor descriptor
        const BITensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
        const bool               is_nhwc           = input_tensor_desc.layout == BIDataLayout::NHWC;

        // Box format: [Objectness:1][Box:4][Classes:N]

        // Activate objectness and front part of the box
        const BICoordinates box_start(0, 0, 0);
        const BICoordinates box_end = is_nhwc ? BICoordinates(3, -1, -1) : BICoordinates(-1, -1, 3);
        NodeID box     = g.add_node<SliceLayerNode>(box_start, box_end);
        NodeID act_box = g.add_node<ActivationLayerNode>(act_info);
        set_node_params(g, box, params);
        set_node_params(g, act_box, params);
        g.add_connection(input.node_id, input.index, box, 0);
        g.add_connection(box, 0, act_box, 0);

        // Immutable part
        const BICoordinates imm_start = is_nhwc ? BICoordinates(3, 0, 0) : BICoordinates(0, 0, 3);
        const BICoordinates imm_end   = is_nhwc ? BICoordinates(5, -1, -1) : BICoordinates(-1, -1, 5);
        NodeID            imm       = g.add_node<SliceLayerNode>(imm_start, imm_end);
        set_node_params(g, imm, params);
        g.add_connection(input.node_id, input.index, imm, 0);

        // Activation classes and end part of box
        const BICoordinates cls_start = is_nhwc ? BICoordinates(5, 0, 0) : BICoordinates(0, 0, 5);
        const BICoordinates cls_end   = BICoordinates(-1, -1, -1);
        NodeID            cls       = g.add_node<SliceLayerNode>(cls_start, cls_end);
        NodeID            cls_act   = g.add_node<ActivationLayerNode>(act_info);
        set_node_params(g, cls, params);
        set_node_params(g, cls_act, params);
        g.add_connection(input.node_id, input.index, cls, 0);
        g.add_connection(cls, 0, cls_act, 0);

        NodeID concat =
            g.add_node<ConcatenateLayerNode>(3, descriptors::ConcatLayerDescriptor(BIDataLayoutDimension::CHANNEL));
        set_node_params(g, concat, params);
        g.add_connection(act_box, 0, concat, 0);
        g.add_connection(imm, 0, concat, 1);
        g.add_connection(cls_act, 0, concat, 2);

        return concat;
    }

} // namespace graph

} // namespace BatmanInfer
