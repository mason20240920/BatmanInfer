//
// Created by holynova on 2024/12/30.
//

#ifndef BATMANINFER_BI_GRAPH_TYPES_H
#define BATMANINFER_BI_GRAPH_TYPES_H

#include "data/core/bi_error.h"
#include "data/core/bi_pixel_value.h"
#include "data/core/bi_types.hpp"
#include "function_info/bi_activationLayerInfo.h"
#include "function_info/bi_convolutionInfo.h"
#include "function_info/bi_fullyConnectedLayerInfo.h"

#include <limits>
#include <string>
#include <map>

namespace BatmanInfer {

namespace graph {

    using BatmanInfer::BIStatus;

    using BatmanInfer::BICoordinates;
    using BatmanInfer::BIDataLayout;
    using BatmanInfer::BIDataLayoutDimension;
    using BatmanInfer::BIDataType;
    using BatmanInfer::BIPixelValue;
    using BatmanInfer::Size2D;
    using BatmanInfer::BITensorShape;

    using BatmanInfer::BIActivationLayerInfo;
    using BatmanInfer::BIDimensionRoundingType;
    using BatmanInfer::BIFullyConnectedLayerInfo;
    using BatmanInfer::BIPadStrideInfo;

    using GraphID    = unsigned int;
    using TensorID   = unsigned int;
    using NodeID     = unsigned int;
    using EdgeID     = unsigned int;
    using Activation = BIActivationLayerInfo::ActivationFunction;

    /**< Constant TensorID specifying an equivalent of null tensor */
    constexpr TensorID NullTensorID = std::numeric_limits<TensorID>::max();
    /**< Constant NodeID specifying an equivalent of null node */
    constexpr NodeID EmptyNodeID = std::numeric_limits<NodeID>::max();
    /**< Constant EdgeID specifying an equivalent of null edge */
    constexpr EdgeID EmptyEdgeID = std::numeric_limits<EdgeID>::max();

    // Forward declarations
    struct BITensorDescriptor;

    /** Graph configuration structure */
    struct BIGraphConfig
    {
        bool        use_function_memory_manager{true};   /**< Use a memory manager to manage per-function auxilary memory */
        bool        use_function_weights_manager{true};  /**< Use a weights manager to manage transformed weights */
        bool        use_transition_memory_manager{true}; /**< Use a memory manager to manager transition buffer memory */
        bool        use_tuner{false};                    /**< Use a tuner in tunable backends */
        bool        use_synthetic_type{false};           /**< Convert graph to a synthetic graph for a data type */
        BIDataType  synthetic_type{BIDataType::QASYMM8};   /**< The data type of the synthetic graph  */
        // CLTunerMode tuner_mode{CLTunerMode::EXHAUSTIVE}; /**< Tuner mode to be used by the CL tuner */
        int         num_threads{
            -1}; /**< Number of threads to use (thread capable backends), if 0 the backend will auto-initialize, if -1 the backend will stay as it is. */
        std::string   tuner_file{"acl_tuner.csv"};         /**< File to load/store tuning values from */
        std::string   mlgo_file{"heuristics.mlgo"};        /**< Filename to load MLGO heuristics from */
        // CLBackendType backend_type{CLBackendType::Native}; /**< CL backend type to use */
    };

    /**< Device target types */
    enum class BITarget
    {
        UNSPECIFIED, /**< Unspecified Target */
        NEON,        /**< Arm® Neon™ capable target device */
        CL,          /**< OpenCL capable target device */
        CLVK,        /**< CLVK capable target device */
    };

    /** Supported Element-wise operations */
    enum class BIEltwiseOperation
    {
        Add, /**< Arithmetic addition */
        Sub, /**< Arithmetic subtraction */
        Mul, /**< Arithmetic multiplication */
        Max, /**< Arithmetic maximum */
        Div, /**< Arithmetic division */
        Min, /**< Arithmetic minimum */
    };

    /** Supported Unary Element-wise operations */
    enum class BIUnaryEltwiseOperation
    {
        Exp /**< Exp */
    };

    /** Supported Convolution layer methods */
    enum class BIConvolutionMethod
    {
        Default, /**< Default approach using internal heuristics */
        GEMM,    /**< GEMM based convolution */
        Direct,  /**< Deep direct convolution */
        Winograd /**< Winograd based convolution */
    };

    /** Supported Depthwise Convolution layer methods */
    enum class BIDepthwiseConvolutionMethod
    {
        Default,      /**< Default approach using internal heuristics */
        GEMV,         /**< Generic GEMV based depthwise convolution */
        Optimized3x3, /**< Optimized 3x3 direct depthwise convolution */
    };

    /** Enable or disable fast math for Convolution layer */
    enum class BIFastMathHint
    {
        Enabled,  /**< Fast math enabled for Convolution layer */
        Disabled, /**< Fast math disabled for Convolution layer */
    };

    /** Supported nodes */
    enum class BINodeType
    {
        ActivationLayer,
        ArgMinMaxLayer,
        BatchNormalizationLayer,
        BoundingBoxTransformLayer,
        ChannelShuffleLayer,
        ConcatenateLayer,
        ConvolutionLayer,
        DeconvolutionLayer,
        DepthToSpaceLayer,
        DepthwiseConvolutionLayer,
        DequantizationLayer,
        DetectionOutputLayer,
        DetectionPostProcessLayer,
        EltwiseLayer,
        FlattenLayer,
        FullyConnectedLayer,
        FusedConvolutionBatchNormalizationLayer,
        FusedDepthwiseConvolutionBatchNormalizationLayer,
        GenerateProposalsLayer,
        L2NormalizeLayer,
        NormalizationLayer,
        NormalizePlanarYUVLayer,
        PadLayer,
        PermuteLayer,
        PoolingLayer,
        PReluLayer,
        PrintLayer,
        PriorBoxLayer,
        QuantizationLayer,
        ReductionOperationLayer,
        ReorgLayer,
        ReshapeLayer,
        ResizeLayer,
        ROIAlignLayer,
        SoftmaxLayer,
        SliceLayer,
        SplitLayer,
        StackLayer,
        StridedSliceLayer,
        UpsampleLayer,
        UnaryEltwiseLayer,

        Input,
        Output,
        Const,

        Dummy
    };

    /** Backend Memory Manager affinity **/
    enum class BIMemoryManagerAffinity
    {
        Buffer, /**< Affinity at buffer level */
        Offset  /**< Affinity at offset level */
    };

    /** NodeID-index struct
     *
     * Used to describe connections
     */
    struct BINodeIdxPair
    {
        NodeID node_id; /**< Node ID */
        size_t index;   /**< Index */
    };

    /** Common node parameters */
    struct BINodeParams
    {
        std::string name;   /**< Node name */
        BITarget    target; /**< Node target */
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_BI_GRAPH_TYPES_H
