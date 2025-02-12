//
// Created by holynova on 2025/2/6.
//

#pragma once

#include "graph/bi_itensorAccessor.h"
#include "graph/bi_layerDescriptors.h"
#include "graph/bi_tensorDescriptor.h"
#include "graph/bi_types.h"

namespace BatmanInfer {

namespace graph {

    // Forward declaration
    class BIGraph;

    /** Graph builder class
     *
     * Builds and compiles a graph
     */
    class GraphBuilder final
    {
    public:
        /** Adds a Const node to the graph
         *
         * @param[in] g        Graph to add the node to
         * @param[in] params   Common node parameters
         * @param[in] desc     Tensor descriptor of the node
         * @param[in] accessor (Optional) Accessor of the const node data
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID
        add_const_node(BIGraph &g, BINodeParams params, const BITensorDescriptor &desc,
                     BIITensorAccessorUPtr accessor = nullptr);

        /** Adds an input layer node to the graph
         *
         * @param[in] g        Graph to add the node to
         * @param[in] params   Common node parameters
         * @param[in] desc     Tensor descriptor of the Tensor
         * @param[in] accessor (Optional) Accessor of the input node data
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID
        add_input_node(BIGraph &g, BINodeParams params, const BITensorDescriptor &desc,
                       BIITensorAccessorUPtr accessor = nullptr);

        /** Adds an output layer node to the graph
         *
         * @param[in] g        Graph to add the node to
         * @param[in] params   Common node parameters
         * @param[in] input    Input to the output node as a NodeID-Index pair
         * @param[in] accessor (Optional) Accessor of the output node data
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID
        add_output_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                        BIITensorAccessorUPtr accessor = nullptr);

        /** Adds an activation layer node to the graph
         *
         * @param[in] g              Graph to add the node to
         * @param[in] params         Common node parameters
         * @param[in] input          Input to the activation layer node as a NodeID-Index pair
         * @param[in] act_info       Activation layer information
         * @param[in] out_quant_info (Optional) Output quantization info
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_activation_node(BIGraph                  &g,
                                          BINodeParams              params,
                                          BINodeIdxPair             input,
                                          BIActivationLayerInfo     act_info,
                                          const BIQuantizationInfo &out_quant_info = BIQuantizationInfo());

        /** Adds an activation layer node to the graph
         *
         * @param[in] g              Graph to add the node to
         * @param[in] params         Common node parameters
         * @param[in] input          Input to the activation layer node as a NodeID-Index pair
         * @param[in] op             Reduction Operation: min or max
         * @param[in] axis           Axis to perform reduction operation across
         * @param[in] out_data_type  (Optional) Output data type
         * @param[in] out_quant_info (Optional) Output quantization info
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_arg_min_max_node(BIGraph                  &g,
                                           BINodeParams              params,
                                           BINodeIdxPair             input,
                                           BIReductionOperation      op,
                                           unsigned int              axis,
                                           BIDataType                out_data_type  = BIDataType::UNKNOWN,
                                           const BIQuantizationInfo &out_quant_info = BIQuantizationInfo());

        /** Adds a batch normalization layer node to the graph
         *
         * @param[in] g              Graph to add the node to
         * @param[in] params         Common node parameters
         * @param[in] input          Input to the batch normalization layer node as a NodeID-Index pair
         * @param[in] epsilon        Epsilon parameter
         * @param[in] mean_accessor  Const Node ID that contains the mean values
         * @param[in] var_accessor   Const Node ID that contains the variance values
         * @param[in] beta_accessor  Const Node ID that contains the beta values. Can be EmptyNodeID
         * @param[in] gamma_accessor Const Node ID that contains the gamma values. Can be EmptyNodeID
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_batch_normalization_node(BIGraph              &g,
                                                   BINodeParams          params,
                                                   BINodeIdxPair         input,
                                                   float                 epsilon,
                                                   BIITensorAccessorUPtr mean_accessor  = nullptr,
                                                   BIITensorAccessorUPtr var_accessor   = nullptr,
                                                   BIITensorAccessorUPtr beta_accessor  = nullptr,
                                                   BIITensorAccessorUPtr gamma_accessor = nullptr);

        /** Adds a bounding box transform layer node to the graph
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] input  Input to the bounding box transform layer node as a NodeID-Index pair
         * @param[in] deltas Deltas input to the bounding box transform layer node as a NodeID-Index pair
         * @param[in] info   Bounding Box Transform information
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_bounding_box_transform_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                                      BINodeIdxPair deltas, BIBoundingBoxTransformInfo info);

        /** Adds an channel shuffle layer node to the graph
         *
         * @param[in] g          Graph to add the node to
         * @param[in] params     Common node parameters
         * @param[in] input      Input to the activation layer node as a NodeID-Index pair
         * @param[in] num_groups Number of groups
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_channel_shuffle_node(
            BIGraph &g, BINodeParams params, BINodeIdxPair input, unsigned int num_groups);

        /** Adds a convolution layer node to the graph
         *
         * @param[in] g                     Graph to add the node to
         * @param[in] params                Common node parameters
         * @param[in] input                 Input to the convolution layer node as a NodeID-Index pair
         * @param[in] kernel_spatial_extend Spatial extend of convolution kernels
         * @param[in] depth                 Number of convolution kernels
         * @param[in] conv_info             Convolution layer information
         * @param[in] num_groups            (Optional) Number of groups for a grouped convolution. Defaults to 1
         * @param[in] method                (Optional) Convolution method to use
         * @param[in] fast_math_hint        (Optional) Fast math hint
         * @param[in] weights_accessor      (Optional) Accessor of the weights node data
         * @param[in] bias_accessor         (Optional) Accessor of the bias node data
         * @param[in] weights_quant_info    (Optional) Weights quantization info
         * @param[in] out_quant_info        (Optional) Output quantization info
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_convolution_node(BIGraph                  &g,
                                           BINodeParams              params,
                                           BINodeIdxPair             input,
                                           Size2D                    kernel_spatial_extend,
                                           unsigned int              depth,
                                           BIPadStrideInfo           conv_info,
                                           unsigned int              num_groups         = 1,
                                           BIConvolutionMethod       method             = BIConvolutionMethod::Default,
                                           BIFastMathHint            fast_math_hint     = BIFastMathHint::Disabled,
                                           BIITensorAccessorUPtr     weights_accessor   = nullptr,
                                           BIITensorAccessorUPtr     bias_accessor      = nullptr,
                                           const BIQuantizationInfo &weights_quant_info = BIQuantizationInfo(),
                                           const BIQuantizationInfo &out_quant_info     = BIQuantizationInfo());

        /** Adds a deconvolution layer node to the graph
         *
         * @param[in] g                     Graph to add the node to
         * @param[in] params                Common node parameters
         * @param[in] input                 Input to the convolution layer node as a NodeID-Index pair
         * @param[in] kernel_spatial_extend Spatial extend of convolution kernels
         * @param[in] depth                 Number of convolution kernels
         * @param[in] deconv_info           Convolution layer information
         * @param[in] weights_accessor      (Optional) Accessor of the weights node data
         * @param[in] bias_accessor         (Optional) Accessor of the bias node data
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_deconvolution_node(BIGraph              &g,
                                             BINodeParams          params,
                                             BINodeIdxPair         input,
                                             Size2D                kernel_spatial_extend,
                                             unsigned int          depth,
                                             BIPadStrideInfo       deconv_info,
                                             BIITensorAccessorUPtr weights_accessor = nullptr,
                                             BIITensorAccessorUPtr bias_accessor    = nullptr);

        /** Adds a depth concatenate node to the graph
         *
         * @param[in] g                 Graph to add the node to
         * @param[in] params            Common node parameters
         * @param[in] inputs            Inputs to the concatenate layer node as a NodeID-Index pair
         * @param[in] concat_descriptor Concatenation layer descriptor
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_concatenate_node(BIGraph                                  &g,
                                           BINodeParams                              params,
                                           const std::vector<BINodeIdxPair>         &inputs,
                                           const descriptors::ConcatLayerDescriptor &concat_descriptor);

        /** Adds an depth to space layer node to the graph
         *
         * @param[in] g           Graph to add the node to
         * @param[in] params      Common node parameters
         * @param[in] input       Input to the depth to space layer node as a NodeID-Index pair
         * @param[in] block_shape Block shape to reshape tensor with
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_depth_to_space_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, int32_t block_shape);

        /** Adds a depth-wise convolution layer node to the graph
         *
         * @param[in] g                     Graph to add the node to
         * @param[in] params                Common node parameters
         * @param[in] input                 Input to the depthwise convolution layer node as a NodeID-Index pair
         * @param[in] kernel_spatial_extend Spatial extend of convolution kernels
         * @param[in] conv_info             Convolution layer information
         * @param[in] depth_multiplier      (Optional) Depth multiplier parameter.
         * @param[in] method                (Optional) Convolution method to use
         * @param[in] weights_accessor      (Optional) Accessor of the weights node data
         * @param[in] bias_accessor         (Optional) Accessor of the bias node data
         * @param[in] quant_info            (Optional) Weights quantization info
         * @param[in] out_quant_info        (Optional) Output quantization info
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID
        add_depthwise_convolution_node(
            BIGraph                     &g,
            BINodeParams                 params,
            BINodeIdxPair                input,
            Size2D                       kernel_spatial_extend,
            BIPadStrideInfo              conv_info,
            int                          depth_multiplier = 1,
            BIDepthwiseConvolutionMethod method           = BIDepthwiseConvolutionMethod::Default,
            BIITensorAccessorUPtr        weights_accessor = nullptr,
            BIITensorAccessorUPtr        bias_accessor    = nullptr,
            const BIQuantizationInfo    &quant_info       = BIQuantizationInfo(),
            const BIQuantizationInfo    &out_quant_info   = BIQuantizationInfo());

        /** Adds an element-wise layer node to the graph
         *
         * @param[in] g         Graph to add the node to
         * @param[in] params    Common node parameters
         * @param[in] input0    First input to the element-wise operation layer node as a NodeID-Index pair
         * @param[in] input1    Second input to the element-wise operation layer node as a NodeID-Index pair
         * @param[in] operation Element-wise operation to perform
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_elementwise_node(
            BIGraph &g, BINodeParams params, BINodeIdxPair input0, BINodeIdxPair input1, BIEltwiseOperation operation);

        /** Adds a dequantization node to the graph
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] input  Input to the dequantization node as a NodeID-Index pair
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_dequantization_node(BIGraph &g, BINodeParams params, BINodeIdxPair input);

        /** Adds a detection output layer node to the graph
         *
         * @param[in] g              Graph to add the node to
         * @param[in] params         Common node parameters
         * @param[in] input_loc      Location input to the detection output layer node as a NodeID-Index pair
         * @param[in] input_conf     Confidence input to the detection output layer node as a NodeID-Index pair
         * @param[in] input_priorbox PriorBox input to the detection output layer node as a NodeID-Index pair
         * @param[in] detect_info    Detection output layer parameters
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_detection_output_node(BIGraph                          &g,
                                                BINodeParams                      params,
                                                BINodeIdxPair                     input_loc,
                                                BINodeIdxPair                     input_conf,
                                                BINodeIdxPair                     input_priorbox,
                                                const BIDetectionOutputLayerInfo &detect_info);

        /** Adds a detection post process layer node to the graph
         *
         * @param[in] g                      Graph to add the node to
         * @param[in] params                 Common node parameters
         * @param[in] input_box_encoding     Boxes input to the detection output layer node as a NodeID-Index pair
         * @param[in] input_class_prediction Class prediction input to the detection output layer node as a NodeID-Index pair
         * @param[in] detect_info            Detection output layer parameters
         * @param[in] anchors_accessor       (Optional) Const Node ID that contains the anchor values
         * @param[in] anchor_quant_info      (Optional) Anchor quantization info
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_detection_post_process_node(
            BIGraph                               &g,
            BINodeParams                           params,
            BINodeIdxPair                          input_box_encoding,
            BINodeIdxPair                          input_class_prediction,
            const BIDetectionPostProcessLayerInfo &detect_info,
            BIITensorAccessorUPtr                  anchors_accessor = nullptr,
            const BIQuantizationInfo &anchor_quant_info = BIQuantizationInfo());

        /** Adds a Dummy node to the graph
         *
         * @note this node if for debugging purposes. Just alters the shape of the graph pipeline as requested.
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] input  Input to the dummy node as a NodeID-Index pair
         * @param[in] shape  Output shape
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_dummy_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BITensorShape shape);

        /** Adds a flatten layer node to the graph
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] input  Input to the flatten layer node as a NodeID-Index pair
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_flatten_node(BIGraph &g, BINodeParams params, BINodeIdxPair input);

        /** Adds a fully connected layer node to the graph
         *
         * @param[in] g              Graph to add the layer to
         * @param[in] params         Common node parameters
         * @param[in] input          Input to the fully connected layer node as a NodeID-Index pair
         * @param[in] num_outputs    Number of output neurons
         * @param[in] weights_nid    Node ID of the weights node data
         * @param[in] bias_nid       (Optional) Node ID of the bias node data. Defaults to EmptyNodeID
         * @param[in] fc_info        (Optional) Fully connected layer metadata
         * @param[in] out_quant_info (Optional) Output quantization info
         * @param[in] fast_math_hint (Optional) Fast math hint
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_fully_connected_layer(
            BIGraph                        &g,
            BINodeParams                    params,
            BINodeIdxPair                   input,
            unsigned int                    num_outputs,
            NodeID                          weights_nid,
            NodeID                          bias_nid       = EmptyNodeID,
            const BIFullyConnectedLayerInfo fc_info        = BIFullyConnectedLayerInfo(),
            const BIQuantizationInfo       &out_quant_info = BIQuantizationInfo(),
            BIFastMathHint                  fast_math_hint = BIFastMathHint::Disabled);

        /** Adds a fully connected layer node to the graph
         *
         * @param[in] g                  Graph to add the layer to
         * @param[in] params             Common node parameters
         * @param[in] input              Input to the fully connected layer node as a NodeID-Index pair
         * @param[in] num_outputs        Number of output neurons
         * @param[in] weights_accessor   (Optional) Accessor of the weights node data
         * @param[in] bias_accessor      (Optional) Accessor of the bias node data
         * @param[in] fc_info            (Optional) Fully connected layer metadata
         * @param[in] weights_quant_info (Optional) Weights quantization info
         * @param[in] out_quant_info     (Optional) Output quantization info
         * @param[in] fast_math_hint     (Optional) Fast math hint
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_fully_connected_layer(
            BIGraph                        &g,
            BINodeParams                    params,
            BINodeIdxPair                   input,
            unsigned int                    num_outputs,
            BIITensorAccessorUPtr           weights_accessor = nullptr,
            BIITensorAccessorUPtr           bias_accessor    = nullptr,
            const BIFullyConnectedLayerInfo fc_info          = BIFullyConnectedLayerInfo(),
            const BIQuantizationInfo       &weights_quant_info = BIQuantizationInfo(),
            const BIQuantizationInfo       &out_quant_info     = BIQuantizationInfo(),
            BIFastMathHint                  fast_math_hint     = BIFastMathHint::Disabled);

        /** Adds a generate proposals layer node to the graph
         *
         * @param[in] g       Graph to add the layer to
         * @param[in] params  Common node parameters
         * @param[in] scores  Input scores to the generate proposals layer node as a NodeID-Index pair
         * @param[in] deltas  Input deltas to the generate proposals layer node as a NodeID-Index pair
         * @param[in] anchors Input anchors to the generate proposals layer node as a NodeID-Index pair
         * @param[in] info    Generate proposals operation information
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_generate_proposals_node(BIGraph                &g,
                                                  BINodeParams            params,
                                                  BINodeIdxPair           scores,
                                                  BINodeIdxPair           deltas,
                                                  BINodeIdxPair           anchors,
                                                  BIGenerateProposalsInfo info);

        /** Adds a L2 Normalize layer node to the graph
         *
         * @param[in] g       Graph to add the node to
         * @param[in] params  Common node parameters
         * @param[in] input   Input to the normalization layer node as a NodeID-Index pair
         * @param[in] axis    Axis to perform normalization on
         * @param[in] epsilon Lower bound value for the normalization
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_l2_normalize_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, int axis, float epsilon);

        /** Adds a normalization layer node to the graph
         *
         * @param[in] g         Graph to add the node to
         * @param[in] params    Common node parameters
         * @param[in] input     Input to the normalization layer node as a NodeID-Index pair
         * @param[in] norm_info Normalization layer information
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID
        add_normalization_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BINormalizationLayerInfo norm_info);

        /** Adds a normalize planar YUV layer node to the graph
         *
         * @param[in] g             Graph to add the node to
         * @param[in] params        Common node parameters
         * @param[in] input         Input to the normalize planar YUV layer node as a NodeID-Index pair
         * @param[in] mean_accessor Const Node ID that contains the mean values
         * @param[in] std_accessor  Const Node ID that contains the variance values
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_normalize_planar_yuv_node(BIGraph              &g,
                                                    BINodeParams          params,
                                                    BINodeIdxPair         input,
                                                    BIITensorAccessorUPtr mean_accessor = nullptr,
                                                    BIITensorAccessorUPtr std_accessor  = nullptr);

        /** Adds a pad layer node to the graph
         *
         * @param[in] g         Graph to add the node to
         * @param[in] params    Common node parameters
         * @param[in] input     Input to the reshape layer node as a NodeID-Index pair
         * @param[in] paddings  The padding for each spatial dimension of the input tensor. The pair padding[i]
         *                      specifies the front and the end padding in the i-th dimension.
         * @param[in] pad_value Padding value to be used. Defaults to 0
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_pad_node(BIGraph           &g,
                                   BINodeParams       params,
                                   BINodeIdxPair      input,
                                   const PaddingList &paddings,
                                   BIPixelValue       pad_value = BIPixelValue());

        /** Adds a permute layer node to the graph
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] input  Input to the reshape layer node as a NodeID-Index pair
         * @param[in] perm   Permutation vector
         * @param[in] layout (Optional) Data layout to assign to permuted tensor.
         *                    If UNKNOWN then the input's layout will be used.
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_permute_node(BIGraph          &g,
                                       BINodeParams      params,
                                       BINodeIdxPair     input,
                                       PermutationVector perm,
                                       BIDataLayout      layout = BIDataLayout::UNKNOWN);

        /** Adds a pooling layer node to the graph
         *
         * @param[in] g         Graph to add the node to
         * @param[in] params    Common node parameters
         * @param[in] input     Input to the pooling layer node as a NodeID-Index pair
         * @param[in] pool_info Pooling layer information
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_pooling_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BIPoolingLayerInfo pool_info);

        /** Adds a prelu layer node to the graph
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] input  Input to the PRelu node as a NodeID-Index pair
         * @param[in] alpha  Alpha input to the PRelu node as a NodeID-Index pair
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_prelu_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BINodeIdxPair alpha);

        /** Adds a print layer node to the graph
         *
         * @param[in] g           Graph to add the node to
         * @param[in] params      Common node parameters
         * @param[in] input       Input to the print layer node as a NodeID-Index pair
         * @param[in] stream      Output stream.
         * @param[in] format_info (Optional) Format info.
         * @param[in] transform   (Optional) Transformation function to be applied to the input tensor before printing.
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_print_node(BIGraph                                      &g,
                                     BINodeParams                                  params,
                                     BINodeIdxPair                                 input,
                                     std::ostream                                 &stream,
                                     const BIIOFormatInfo                         &format_info = BIIOFormatInfo(),
                                     const std::function<BIITensor *(BIITensor *)> transform   = nullptr);

        /** Adds a priorbox layer node to the graph
         *
         * @param[in] g          Graph to add the node to
         * @param[in] params     Common node parameters
         * @param[in] input0     First input to the priorbox layer node as a NodeID-Index pair
         * @param[in] input1     Second input to the priorbox layer node as a NodeID-Index pair
         * @param[in] prior_info PriorBox parameters
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_priorbox_node(BIGraph &g, BINodeParams params, BINodeIdxPair input0, BINodeIdxPair input1,
                                        const BIPriorBoxLayerInfo &prior_info);

        /** Adds a quantization layer node to the graph
         *
         * @param[in] g              Graph to add the node to
         * @param[in] params         Common node parameters
         * @param[in] input          Input to the quantization layer node as a NodeID-Index pair
         * @param[in] out_quant_info Output quantization info
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID
        add_quantization_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                              const BIQuantizationInfo &out_quant_info);

        /** Adds a reduction sum layer node to the graph
         *
         * @param[in] g         Graph to add the node to
         * @param[in] params    Common node parameters
         * @param[in] input     Input to the reorg layer node as a NodeID-Index pair
         * @param[in] op        Reduction operation
         * @param[in] axis      Reduction axis
         * @param[in] keep_dims (Optional) Whether to keep the reduced dimension after the operation. Defaults to true.
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_reduction_operation_node(BIGraph &g, BINodeParams params, BINodeIdxPair input,
                                                   BIReductionOperation op, int axis, bool keep_dims = true);

        /** Adds a reorg layer node to the graph
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] input  Input to the reorg layer node as a NodeID-Index pair
         * @param[in] stride Stride value to use for reorganizing the values in the output tensor.
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_reorg_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, int stride);

        /** Adds a reshape layer node to the graph
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] input  Input to the reshape layer node as a NodeID-Index pair
         * @param[in] shape  Output reshaped shape
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_reshape_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BITensorShape shape);

        /** Adds a resize layer node to the graph
         *
         * @param[in] g            Graph to add the node to
         * @param[in] params       Common node parameters
         * @param[in] input        Input to the reshape layer node as a NodeID-Index pair
         * @param[in] policy       Interpolation policy
         * @param[in] width_scale  Width scaling factor
         * @param[in] height_scale Height scaling factor
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_resize_node(BIGraph              &g,
                                      BINodeParams          params,
                                      BINodeIdxPair         input,
                                      BIInterpolationPolicy policy,
                                      float                 width_scale,
                                      float                 height_scale);

        /** Adds a ROI align layer node to the graph
         *
         * @param[in] g         Graph to add the node to
         * @param[in] params    Common node parameters
         * @param[in] input     Input to the reshape layer node as a NodeID-Index pair
         * @param[in] rois      Input containing the ROIs.
         * @param[in] pool_info Contains pooling operation information described in @ref ROIPoolingLayerInfo.
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_roi_align_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BINodeIdxPair rois,
                                         BIROIPoolingLayerInfo pool_info);

        /** Adds a scale layer node to the graph
         * This layer computes a product of the input with a scale (read from mul_accessor) and it applies an offset (read from add_accessor).
         * output = input * mul_w + add_w
         *
         * @param[in] g            Graph to add the layer to
         * @param[in] params       Common node parameters
         * @param[in] input        Input to the fully connected layer node as a NodeID-Index pair
         * @param[in] mul_accessor (Optional) Accessor of the mul node data
         * @param[in] add_accessor (Optional) Accessor of the add node data
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_scale_layer(BIGraph              &g,
                                      const BINodeParams   &params,
                                      BINodeIdxPair         input,
                                      BIITensorAccessorUPtr mul_accessor = nullptr,
                                      BIITensorAccessorUPtr add_accessor = nullptr);

        /** Adds a softmax node to the graph
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] input  Input to the softmax layer node as a NodeID-Index pair
         * @param[in] beta   Beta parameter
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_softmax_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, float beta = 1.f);

        /** Adds a slice node to the graph
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] input  Input to the slice layer node as a NodeID-Index pair
         * @param[in] starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_slice_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BICoordinates &starts,
                                     BICoordinates &ends);

        /** Adds a split node to the graph
         *
         * @param[in] g          Graph to add the node to
         * @param[in] params     Common node parameters
         * @param[in] input      Input to the split layer node as a NodeID-Index pair
         * @param[in] num_splits Number of different splits
         * @param[in] axis       (Optional) Split axis. Defaults to 0
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_split_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, unsigned int num_splits,
                                     unsigned int axis = 0);

        /** Adds a stack layer node to the graph
         *
         * @param[in] g      Graph to add the node to
         * @param[in] params Common node parameters
         * @param[in] inputs Inputs to the reorg layer node as a NodeID-Index pair
         * @param[in] axis   Axis along which the input tensors have to be packed
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_stack_node(BIGraph &g, BINodeParams params, const std::vector<BINodeIdxPair> &inputs, int axis);

        /** Adds a strided slice node to the graph
         *
         * @param[in] g       Graph to add the node to
         * @param[in] params  Common node parameters
         * @param[in] input   Input to the strided slice layer node as a NodeID-Index pair
         * @param[in] starts  The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] ends    The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] strides The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] info    Contains masks for the starts, ends and strides
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_strided_slice_node(BIGraph                &g,
                                             BINodeParams            params,
                                             BINodeIdxPair           input,
                                             BICoordinates          &starts,
                                             BICoordinates          &ends,
                                             BiStrides              &strides,
                                             BIStridedSliceLayerInfo info);

        /** Adds a yolo layer to the graph
         *
         * @param[in] g        Graph to add the node to
         * @param[in] params   Common node parameters
         * @param[in] input    Input to the yolo layer node as a NodeID-Index pair
         * @param[in] act_info Activation layer parameters
         *
         * @return Node ID of the created node, EmptyNodeID in case of error
         */
        static NodeID add_yolo_node(BIGraph &g, BINodeParams params, BINodeIdxPair input, BIActivationLayerInfo act_info);

    };

} // namespace graph

} // namespace BatmanInfer
