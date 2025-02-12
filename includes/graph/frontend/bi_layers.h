//
// Created by holynova on 2025/2/6.
//

#pragma once

#include "data/core/utils/misc/utils.hpp"
#include "graph/frontend/bi_ilayer.h"
#include "graph/frontend/bi_igraph_front.h"
#include "graph/bi_graphBuilder.h"
#include "graph/bi_types.h"

#include <memory>
#include <string>

namespace BatmanInfer {

namespace graph {

namespace frontend {

    /** Input Layer */
    class BIInputLayer final : public BIILayer
    {
    public:
        /** Construct an input layer.
         *
         * @param[in] desc     Description of input tensor.
         * @param[in] accessor Accessor to get input tensor data from.
         */
        BIInputLayer(BITensorDescriptor desc, BIITensorAccessorUPtr accessor)
            : _desc(desc), _accessor(std::move(accessor))
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams common_params = {name(), s.hints().target_hint};
            return GraphBuilder::add_input_node(s.graph(), common_params, _desc, std::move(_accessor));
        }

    private:
        BITensorDescriptor    _desc;
        BIITensorAccessorUPtr _accessor;
    };

    /** Constant Layer */
    class BIConstantLayer final : public BIILayer
    {
    public:
        /** Construct a constant layer.
         *
         * @param[in] desc     Description of input tensor.
         * @param[in] accessor Accessor to get input tensor data from.
         */
        BIConstantLayer(BITensorDescriptor desc, BIITensorAccessorUPtr accessor)
            : _desc(desc), _accessor(std::move(accessor))
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams common_params = {name(), s.hints().target_hint};
            return GraphBuilder::add_const_node(s.graph(), common_params, _desc, std::move(_accessor));
        }

    private:
        BITensorDescriptor    _desc;
        BIITensorAccessorUPtr _accessor;
    };

    /** Output Layer */
    class BIOutputLayer final : public BIILayer
    {
    public:
        /** Construct an output layer.
         *
         * @param[in] accessor       Accessor to give output tensor data to.
         * @param[in] connection_idx (Optional) Input connection index
         */
        BIOutputLayer(BIITensorAccessorUPtr accessor, unsigned int connection_idx = 0)
            : _accessor(std::move(accessor)), _connection_idx(connection_idx)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), _connection_idx};
            return GraphBuilder::add_output_node(s.graph(), common_params, input, std::move(_accessor));
        }

    private:
        BIITensorAccessorUPtr _accessor;
        unsigned int          _connection_idx;
    };

    /** Activation Layer */
    class BIActivationLayer final : public BIILayer
    {
    public:
        /** Construct an activation layer.
         *
         * @param[in] act_info       Activation information
         * @param[in] out_quant_info (Optional) Output quantization info
         */
        BIActivationLayer(BIActivationLayerInfo act_info, const BIQuantizationInfo out_quant_info = BIQuantizationInfo())
            : _act_info(act_info), _out_quant_info(std::move(out_quant_info))
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_activation_node(s.graph(), common_params, input, _act_info,
                                                     std::move(_out_quant_info));
        }

    private:
        BIActivationLayerInfo    _act_info;
        const BIQuantizationInfo _out_quant_info;
    };

    /** ArgMinMax Layer */
    class BIArgMinMaxLayer final : public BIILayer
    {
    public:
        /** Construct an activation layer.
         *
         * @param[in] op             Reduction Operation: min or max
         * @param[in] axis           Axis to perform reduction along
         * @param[in] out_data_type  (Optional) Output tensor data type
         * @param[in] out_quant_info (Optional) Output quantization info
         */
        BIArgMinMaxLayer(BIReductionOperation     op,
                         unsigned int           axis,
                         BIDataType               out_data_type  = BIDataType::UNKNOWN,
                         const BIQuantizationInfo out_quant_info = BIQuantizationInfo())
            : _op(op), _axis(axis), _out_data_type(out_data_type), _out_quant_info(std::move(out_quant_info))
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_arg_min_max_node(s.graph(), common_params, input, _op, _axis, _out_data_type,
                                                      std::move(_out_quant_info));
        }

    private:
        BIReductionOperation _op;
        unsigned int         _axis;
        BIDataType           _out_data_type;
        BIQuantizationInfo   _out_quant_info;
    };

    /** Batchnormalization Layer */
    class BIBatchNormalizationLayer final : public BIILayer
    {
    public:
        /** Construct a batch normalization layer.
         *
         * @param[in] mean    Accessor to get mean tensor data from.
         * @param[in] var     Accessor to get var tensor data from.
         * @param[in] gamma   (Optional) Accessor to get gamma tensor data from. Default: nullptr.
         * @param[in] beta    (Optional) Accessor to get beta tensor data from. Default: nullptr.
         * @param[in] epsilon (Optional) Epsilon value. Default: 0.001.
         */
        BIBatchNormalizationLayer(BIITensorAccessorUPtr mean,
                                  BIITensorAccessorUPtr var,
                                  BIITensorAccessorUPtr gamma   = nullptr,
                                  BIITensorAccessorUPtr beta    = nullptr,
                                  float                 epsilon = 0.001f)
            : _mean(std::move(mean)),
              _var(std::move(var)),
              _gamma(std::move(gamma)),
              _beta(std::move(beta)),
              _epsilon(epsilon)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BI_COMPUTE_ERROR_ON(_mean == nullptr);
            BI_COMPUTE_ERROR_ON(_var == nullptr);

            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_batch_normalization_node(s.graph(), common_params, input, _epsilon, std::move(_mean),
                                                              std::move(_var), std::move(_beta), std::move(_gamma));
        }

    private:
        BIITensorAccessorUPtr _mean;
        BIITensorAccessorUPtr _var;
        BIITensorAccessorUPtr _gamma;
        BIITensorAccessorUPtr _beta;
        float                 _epsilon;
    };

    /** Bounding Box Transform Layer */
    // class BIBoundingBoxTransformLayer final : public BIILayer
    // {
    // };

    /** Channel Shuffle Layer */
    class BIChannelShuffleLayer final : public BIILayer
    {
    public:
        /** Construct a Channel Shuffle layer.
         *
         * @param[in] num_groups Number of groups
         */
        BIChannelShuffleLayer(unsigned int num_groups) : _num_groups(num_groups)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_channel_shuffle_node(s.graph(), common_params, input, _num_groups);
        }

    private:
        unsigned int _num_groups;
    };

    /** Concat Layer */
    // class BIConcatLayer final : public BIILayer
    // {};

    /** Convolution Layer */
    class BIConvolutionLayer final : public BIILayer
    {
    public:
        /** Construct a convolution layer.
         *
         * @param[in] conv_width         Convolution width.
         * @param[in] conv_height        Convolution height.
         * @param[in] ofm                Output feature map.
         * @param[in] weights            Accessor to get kernel weights from.
         * @param[in] bias               Accessor to get kernel bias from.
         * @param[in] conv_info          Padding and stride information.
         * @param[in] num_groups         (Optional) Number of groups. Default: 1.
         * @param[in] weights_quant_info (Optional) Weights quantization information
         * @param[in] out_quant_info     (Optional) Output quantization info
         */
        BIConvolutionLayer(unsigned int             conv_width,
                           unsigned int             conv_height,
                           unsigned int             ofm,
                           BIITensorAccessorUPtr    weights,
                           BIITensorAccessorUPtr    bias,
                           BIPadStrideInfo          conv_info,
                           unsigned int             num_groups         = 1,
                           const BIQuantizationInfo weights_quant_info = BIQuantizationInfo(),
                           const BIQuantizationInfo out_quant_info     = BIQuantizationInfo())
            : _conv_width(conv_width),
              _conv_height(conv_height),
              _ofm(ofm),
              _conv_info(std::move(conv_info)),
              _num_groups(num_groups),
              _weights(std::move(weights)),
              _bias(std::move(bias)),
              _weights_quant_info(std::move(weights_quant_info)),
              _out_quant_info(std::move(out_quant_info))
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeIdxPair input         = {s.tail_node(), 0};
            BINodeParams  common_params = {name(), s.hints().target_hint};
            return GraphBuilder::add_convolution_node(s.graph(), common_params, input, Size2D(_conv_width, _conv_height),
                                                      _ofm, _conv_info, _num_groups, s.hints().convolution_method_hint,
                                                      s.hints().fast_math_hint, std::move(_weights), std::move(_bias),
                                                      std::move(_weights_quant_info), std::move(_out_quant_info));
        }

    private:
        unsigned int             _conv_width;
        unsigned int             _conv_height;
        unsigned int             _ofm;
        const BIPadStrideInfo    _conv_info;
        unsigned int             _num_groups;
        BIITensorAccessorUPtr    _weights;
        BIITensorAccessorUPtr    _bias;
        const BIQuantizationInfo _weights_quant_info;
        const BIQuantizationInfo _out_quant_info;
    };

    /** Deconvolution Layer */
    class BIDeconvolutionLayer final : public BIILayer
    {
    public:
        /** Construct a convolution layer.
         *
         * @param[in] conv_width  Convolution width.
         * @param[in] conv_height Convolution height.
         * @param[in] ofm         Output feature map.
         * @param[in] weights     Accessor to get kernel weights from.
         * @param[in] bias        Accessor to get kernel bias from.
         * @param[in] deconv_info Padding and stride information.
         */
        BIDeconvolutionLayer(unsigned int        conv_width,
                             unsigned int        conv_height,
                             unsigned int        ofm,
                             BIITensorAccessorUPtr weights,
                             BIITensorAccessorUPtr bias,
                             BIPadStrideInfo       deconv_info)
            : _conv_width(conv_width),
              _conv_height(conv_height),
              _ofm(ofm),
              _deconv_info(std::move(deconv_info)),
              _weights(std::move(weights)),
              _bias(std::move(bias))
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeIdxPair input         = {s.tail_node(), 0};
            BINodeParams  common_params = {name(), s.hints().target_hint};
            return GraphBuilder::add_deconvolution_node(s.graph(), common_params, input, Size2D(_conv_width, _conv_height),
                                                        _ofm, _deconv_info, std::move(_weights), std::move(_bias));
        }

    private:
        unsigned int          _conv_width;
        unsigned int          _conv_height;
        unsigned int          _ofm;
        const BIPadStrideInfo _deconv_info;
        BIITensorAccessorUPtr _weights;
        BIITensorAccessorUPtr _bias;
    };

    /** Depthwise Convolution Layer */
    class BIDepthwiseConvolutionLayer final : public BIILayer
    {
    public:
        /** Construct a depthwise convolution layer.
         *
         * @param[in] conv_width         Convolution width.
         * @param[in] conv_height        Convolution height.
         * @param[in] weights            Accessor to get kernel weights from.
         * @param[in] bias               Accessor to get kernel bias from.
         * @param[in] conv_info          Padding and stride information.
         * @param[in] depth_multiplier   (Optional) Depth multiplier parameter.
         * @param[in] weights_quant_info (Optional) Quantization info used for weights
         * @param[in] out_quant_info     (Optional) Output quantization info
         */
        BIDepthwiseConvolutionLayer(unsigned int             conv_width,
                                    unsigned int             conv_height,
                                    BIITensorAccessorUPtr    weights,
                                    BIITensorAccessorUPtr    bias,
                                    BIPadStrideInfo          conv_info,
                                    int                      depth_multiplier   = 1,
                                    const BIQuantizationInfo weights_quant_info = BIQuantizationInfo(),
                                    const BIQuantizationInfo out_quant_info     = BIQuantizationInfo())
            : _conv_width(conv_width),
              _conv_height(conv_height),
              _conv_info(std::move(conv_info)),
              _weights(std::move(weights)),
              _bias(std::move(bias)),
              _depth_multiplier(depth_multiplier),
              _weights_quant_info(std::move(weights_quant_info)),
              _out_quant_info(std::move(out_quant_info))
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeIdxPair input         = {s.tail_node(), 0};
            BINodeParams  common_params = {name(), s.hints().target_hint};
            return GraphBuilder::add_depthwise_convolution_node(
                s.graph(), common_params, input, Size2D(_conv_width, _conv_height), _conv_info, _depth_multiplier,
                s.hints().depthwise_convolution_method_hint, std::move(_weights), std::move(_bias),
                std::move(_weights_quant_info), std::move(_out_quant_info));
        }

    private:
        unsigned int             _conv_width;
        unsigned int             _conv_height;
        const BIPadStrideInfo    _conv_info;
        BIITensorAccessorUPtr    _weights;
        BIITensorAccessorUPtr    _bias;
        int                      _depth_multiplier;
        const BIQuantizationInfo _weights_quant_info;
        const BIQuantizationInfo _out_quant_info;
    };

    /** DepthToSpace Layer */
    class BIDepthToSpaceLayer final : public BIILayer
    {
    public:
        /** Construct an DepthToSpace layer.
         *
         * @param[in] block_shape Block size to rearranged
         */
        BIDepthToSpaceLayer(int32_t block_shape) : _block_shape(block_shape)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_depth_to_space_node(s.graph(), common_params, input, _block_shape);
        }

    private:
        int32_t _block_shape;
    };

    /** Dequantization Layer */
    class BIDequantizationLayer final : public BIILayer
    {
    public:
        /** Construct a dequantization layer.
         *
         */
        BIDequantizationLayer()
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_dequantization_node(s.graph(), common_params, input);
        }
    };

    /** DetectionOutput Layer */
    // class BIDetectionOutputLayer final : public BIILayer
    // {
    // };

    /** DetectionOutputPostProcess Layer */
    // class BIDetectionPostProcessLayer final : public BIILayer
    // {};

    /** Dummy Layer */
    class BIDummyLayer final : public BIILayer
    {
    public:
        /** Construct a dummy layer.
         *
         * @param[in] shape Output shape
         */
        BIDummyLayer(BITensorShape shape) : _shape(shape)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_dummy_node(s.graph(), common_params, input, _shape);
        }

    private:
        BITensorShape _shape;
    };

    // class BIEltwiseLayer final : public BIILayer
    // {};

    /** Flatten Layer */
    class BIFlattenLayer final : public BIILayer
    {
    public:
        /** Construct a flatten layer. */
        BIFlattenLayer()
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_flatten_node(s.graph(), common_params, input);
        }
    };

    /** Fully Connected Layer */
    class BIFullyConnectedLayer final : public BIILayer
    {
    public:
        /** Construct a fully connected layer.
         *
         * @param[in] num_outputs        Number of outputs.
         * @param[in] weights            Accessor to get weights from.
         * @param[in] bias               Accessor to get bias from.
         * @param[in] fc_info            (Optional) Fully connected layer metadata
         * @param[in] weights_quant_info (Optional) Weights quantization information
         * @param[in] out_quant_info     (Optional) Output quantization info
         */
        BIFullyConnectedLayer(unsigned int                    num_outputs,
                              BIITensorAccessorUPtr           weights,
                              BIITensorAccessorUPtr           bias,
                              const BIFullyConnectedLayerInfo fc_info            = BIFullyConnectedLayerInfo(),
                              const BIQuantizationInfo        weights_quant_info = BIQuantizationInfo(),
                              const BIQuantizationInfo        out_quant_info     = BIQuantizationInfo())
            : _num_outputs(num_outputs),
              _weights(std::move(weights)),
              _bias(std::move(bias)),
              // _weights_ss(nullptr),
              // _bias_ss(nullptr),
              _fc_info(fc_info),
              _weights_quant_info(std::move(weights_quant_info)),
              _out_quant_info(std::move(out_quant_info))
        {
        }

        /** Create layer and add to the given stream.
         *
         * @param[in] s Stream to add layer to.
         *
         * @return ID of the created node.
         */
        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            // if (_weights != nullptr)
            // {
                return GraphBuilder::add_fully_connected_layer(
                    s.graph(), common_params, input, _num_outputs, std::move(_weights), std::move(_bias), _fc_info,
                    std::move(_weights_quant_info), std::move(_out_quant_info), s.hints().fast_math_hint);
            // }
            // else
            // {
            //     BI_COMPUTE_ERROR_ON(_weights_ss == nullptr);
            //     NodeID bias_nid = (_bias_ss == nullptr) ? EmptyNodeID : _bias_ss->tail_node();
            //     return GraphBuilder::add_fully_connected_layer(s.graph(), common_params, input, _num_outputs,
            //                                                    _weights_ss->tail_node(), bias_nid, _fc_info,
            //                                                    std::move(_out_quant_info), s.hints().fast_math_hint);
            // }
        }

    private:
        unsigned int                    _num_outputs;
        BIITensorAccessorUPtr           _weights;
        BIITensorAccessorUPtr           _bias;
        // std::unique_ptr<SubStream>    _weights_ss;
        // std::unique_ptr<SubStream>    _bias_ss;
        const BIFullyConnectedLayerInfo _fc_info;
        const BIQuantizationInfo        _weights_quant_info;
        const BIQuantizationInfo        _out_quant_info;
    };

    /** Generate Proposals Layer */
    // class BIGenerateProposalsLayer final : public BIILayer
    // {};

    /** L2 Normalize Layer */
    class BIL2NormalizeLayer final : public BIILayer
    {
    public:
        /** Construct a L2 Normalize layer.
         *
         * @param[in] axis    Axis to perform normalization on
         * @param[in] epsilon Lower bound value for the normalization
         */
        BIL2NormalizeLayer(int axis, float epsilon) : _axis(axis), _epsilon(epsilon)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_l2_normalize_node(s.graph(), common_params, input, _axis, _epsilon);
        }

    private:
        int   _axis;
        float _epsilon;
    };

    /** Normalization Layer */
    class BINormalizationLayer final : public BIILayer
    {
    public:
        /** Construct a normalization layer.
         *
         * @param[in] norm_info Normalization information.
         */
        BINormalizationLayer(BINormalizationLayerInfo norm_info) : _norm_info(norm_info)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_normalization_node(s.graph(), common_params, input, _norm_info);
        }

    private:
        BINormalizationLayerInfo _norm_info;
    };

    /** Normalize planar YUV Layer */
    class BINormalizePlanarYUVLayer final : public BIILayer
    {
    public:
        /** Construct a normalize planar YUV layer.
         *
         * @param[in] mean Accessor to get mean tensor data from.
         * @param[in] std  Accessor to get std tensor data from.
         */
        BINormalizePlanarYUVLayer(BIITensorAccessorUPtr mean, BIITensorAccessorUPtr std)
            : _mean(std::move(mean)), _std(std::move(std))
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BI_COMPUTE_ERROR_ON(_mean == nullptr);
            BI_COMPUTE_ERROR_ON(_std == nullptr);

            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_normalize_planar_yuv_node(s.graph(), common_params, input, std::move(_mean),
                                                               std::move(_std));
        }

    private:
        BIITensorAccessorUPtr _mean;
        BIITensorAccessorUPtr _std;
    };

    /** Pad Layer */
    class BIPadLayer final : public BIILayer
    {
    public:
        /** Construct a pad layer.
         *
         * @param[in] padding   The padding for each spatial dimension of the input tensor. The pair padding[i]
         *                      specifies the front and the end padding in the i-th dimension.
         * @param[in] pad_value Padding value to use. Defaults to 0.
         */
        BIPadLayer(PaddingList padding, BIPixelValue pad_value = BIPixelValue()) : _padding(padding), _pad_value(pad_value)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_pad_node(s.graph(), common_params, input, _padding, _pad_value);
        }

    private:
        PaddingList  _padding;
        BIPixelValue _pad_value;
    };

    /** Permute Layer */
    class BIPermuteLayer final : public BIILayer
    {
    public:
        /** Construct a permute layer.
         *
         * @param[in] perm   Permutation vector.
         * @param[in] layout (Optional) Data layout to assign to permuted tensor.
         *                   If UNKNOWN then the input's layout will be used.
         */
        BIPermuteLayer(PermutationVector perm, BIDataLayout layout = BIDataLayout::UNKNOWN) : _perm(perm), _layout(layout)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_permute_node(s.graph(), common_params, input, _perm, _layout);
        }

    private:
        PermutationVector _perm;
        BIDataLayout      _layout;
    };

    /** Pooling Layer */
    class BIPoolingLayer final : public BIILayer
    {
    public:
        /** Construct a pooling layer.
         *
         * @param[in] pool_info Pooling information.
         */
        BIPoolingLayer(BIPoolingLayerInfo pool_info) : _pool_info(pool_info)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_pooling_node(s.graph(), common_params, input, _pool_info);
        }

    private:
        BIPoolingLayerInfo _pool_info;
    };

    /** PRelu Layer */
    // class BIPReluLayer final : public BIILayer
    // {};

    /** Print Layer */
    class BIPrintLayer final : public BIILayer
    {
    public:
        /** Construct a print layer.
         *
         * Example usage to locally dequantize and print a tensor:
         *
         * Tensor *output = new Tensor();
         * const auto transform = [output](ITensor *input)
         * {
         *     output->allocator()->init(*input->info());
         *     output->info()->set_data_type(DataType::F32);
         *     output->allocator()->allocate();
         *
         *     Window win;
         *     win.use_tensor_dimensions(input->info()->tensor_shape());
         *     Iterator in(input, win);
         *     Iterator out(output, win);
         *     execute_window_loop(win, [&](const Coordinates &)
         *     {
         *         *(reinterpret_cast<float *>(out.ptr())) = dequantize_qasymm8(*in.ptr(), input->info()->quantization_info().uniform());
         *     }, in, out);
         *
         *     return output;
         * };
         *
         * graph << InputLayer(input_descriptor.set_quantization_info(in_quant_info), get_input_accessor(common_params, nullptr, false))
         *       << ...
         *       << \\ CNN Layers
         *       << ...
         *       << PrintLayer(std::cout, IOFormatInfo(), transform)
         *       << ...
         *       << OutputLayer(get_output_accessor(common_params, 5));
         *
         * @param[in] stream      Output stream.
         * @param[in] format_info (Optional) Format info.
         * @param[in] transform   (Optional) Input transform function.
         */
        BIPrintLayer(std::ostream                                 &stream,
                     const BIIOFormatInfo                         &format_info = BIIOFormatInfo(),
                     const std::function<BIITensor *(BIITensor *)> transform   = nullptr)
            : _stream(stream), _format_info(format_info), _transform(transform)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_print_node(s.graph(), common_params, input, _stream, _format_info, _transform);
        }

    private:
        std::ostream                                 &_stream;
        const BIIOFormatInfo                         &_format_info;
        const std::function<BIITensor *(BIITensor *)> _transform;
    };

    /** PriorBox Layer */
    // class BIPriorBoxLayer final : public BIILayer
    // {};

    /** Quantization Layer */
    class BIQuantizationLayer final : public BIILayer
    {
    public:
        /** Construct a quantization layer.
         *
         * @param[in] out_quant_info Output tensor quantization info
         */
        BIQuantizationLayer(BIQuantizationInfo out_quant_info) : _out_quant_info(out_quant_info)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_quantization_node(s.graph(), common_params, input, _out_quant_info);
        }

    private:
        BIQuantizationInfo _out_quant_info;
    };

    /** Reduction Layer */
    class BIReductionLayer final : public BIILayer
    {
    public:
        /** Construct a reduction layer.
         *
         * @param[in] op        Reduction operation
         * @param[in] axis      Reduction axis
         * @param[in] keep_dims (Optional) Whether to keep the reduced dimension after the operation. Defaults to true.
         */
        BIReductionLayer(BIReductionOperation op, unsigned int axis, bool keep_dims)
            : _op(op), _axis(axis), _keep_dims(keep_dims)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_reduction_operation_node(s.graph(), common_params, input, _op, _axis, _keep_dims);
        }

    private:
        BIReductionOperation _op;
        unsigned int         _axis;
        bool                 _keep_dims;
    };

    /** Reorg Layer */
    class BIReorgLayer final : public BIILayer
    {
    public:
        /** Construct a reorg layer.
         *
         * @param[in] stride Stride value to use for reorganizing the values in the output tensor.
         *                   It defines the spatial distance between 2 consecutive pixels in the x and y direction
         */
        BIReorgLayer(int stride) : _stride(stride)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_reorg_node(s.graph(), common_params, input, _stride);
        }

    private:
        int _stride;
    };

    /** Reshape Layer */
    class BIReshapeLayer final : public BIILayer
    {
    public:
        /** Construct a reshape layer.
         *
         * @param[in] shape Target shape.
         */
        BIReshapeLayer(BITensorShape shape) : _shape(shape)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_reshape_node(s.graph(), common_params, input, _shape);
        }

    private:
        BITensorShape _shape;
    };

    /** Resize Layer */
    class BIResizeLayer final : public BIILayer
    {
    public:
        BIResizeLayer(BIInterpolationPolicy policy, float width_scale, float height_scale)
            : _policy(policy), _width_scale(width_scale), _height_scale(height_scale)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_resize_node(s.graph(), common_params, input, _policy, _width_scale, _height_scale);
        }

    private:
        BIInterpolationPolicy _policy;
        float                 _width_scale;
        float                 _height_scale;
    };

    /** ROIAlign Layer */
    // class BIROIAlignLayer final : public BIILayer
    // {};

    /** Scale Layer */
    class BIScaleLayer final : public BIILayer
    {
    public:
        /** Construct a scale layer.
         *
         * @param[in] mul_w Accessor to get mul weight from.
         * @param[in] add_w Accessor to get add weight from.
         */
        BIScaleLayer(BIITensorAccessorUPtr mul_w, BIITensorAccessorUPtr add_w)
            : _mul_w(std::move(mul_w)), _add_w(std::move(add_w))
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_scale_layer(s.graph(), common_params, input, std::move(_mul_w), std::move(_add_w));
        }

    private:
        BIITensorAccessorUPtr _mul_w;
        BIITensorAccessorUPtr _add_w;
    };

    /** Slice Layer */
    class BISliceLayer final : public BIILayer
    {
    public:
        /** Construct a slice layer.
         *
         * @param[in] starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         */
        BISliceLayer(BICoordinates &starts, BICoordinates &ends) : _starts(starts), _ends(ends)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_slice_node(s.graph(), common_params, input, _starts, _ends);
        }

    private:
        BICoordinates _starts;
        BICoordinates _ends;
    };

    /** Softmax Layer */
    class BISoftmaxLayer final : public BIILayer
    {
    public:
        /** Construct a softmax layer.
         *
         * @param[in] beta (Optional) Beta value. Default 1.0.
         */
        BISoftmaxLayer(float beta = 1.0f) : _beta(beta)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_softmax_node(s.graph(), common_params, input, _beta);
        }

    private:
        float _beta;
    };

    /** Stack Layer */
    // class BIStackLayer final : public BIILayer
    // {};

    /** StridedSlice Layer */
    class BIStridedSliceLayer final : public BIILayer
    {
    public:
        /** Construct a strided slice layer.
         *
         * @param[in] starts             The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] ends               The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] strides            The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
         * @param[in] strided_slice_info Contains masks for the starts, ends and strides
         */
        BIStridedSliceLayer(BICoordinates          &starts,
                            BICoordinates          &ends,
                            BiStrides              &strides,
                            BIStridedSliceLayerInfo strided_slice_info)
            : _starts(starts), _ends(ends), _strides(strides), _info(strided_slice_info)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_strided_slice_node(s.graph(), common_params, input, _starts, _ends, _strides, _info);
        }

    private:
        BICoordinates           _starts;
        BICoordinates           _ends;
        BiStrides               _strides;
        BIStridedSliceLayerInfo _info;
    };

    /** YOLO Layer */
    class BIYOLOLayer final : public BIILayer
    {
    public:
        /** Construct a YOLO layer.
         *
         * @param[in] act_info Activation info
         */
        BIYOLOLayer(BIActivationLayerInfo act_info) : _act_info(act_info)
        {
        }

        NodeID create_layer(BIIGraphFront &s) override
        {
            BINodeParams  common_params = {name(), s.hints().target_hint};
            BINodeIdxPair input         = {s.tail_node(), 0};
            return GraphBuilder::add_yolo_node(s.graph(), common_params, input, _act_info);
        }

    private:
        BIActivationLayerInfo _act_info;
    };

} // namespace frontend

} // namespace graph

} // namespace BatmanInfer
