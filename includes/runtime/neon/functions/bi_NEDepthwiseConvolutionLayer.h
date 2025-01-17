//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "runtime/bi_i_memory_manager.hpp"
#include "runtime/bi_memory_group.hpp"
#include "runtime/bi_memory_manager_on_demand.hpp"
#include "runtime/neon/functions/bi_NEActivationLayer.h"
#include "runtime/neon/functions/bi_ne_permute.h"

#include <memory>

namespace BatmanInfer {

    // Forward declarations
    class BIITensor;
    class BINEDepthwiseConvolutionLayerNativeKernel;

    /** Function to execute a depthwise convolution.
     */
    class BINEDepthwiseConvolutionLayer : public BIIFunction
    {
    public:
        /** Default constructor */
        BINEDepthwiseConvolutionLayer(std::shared_ptr<BIIMemoryManager> memory_manager);
        BINEDepthwiseConvolutionLayer() : BINEDepthwiseConvolutionLayer(BIMemoryManagerOnDemand::make_default())
        {
        }
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEDepthwiseConvolutionLayer(const BINEDepthwiseConvolutionLayer &) = delete;
        /** Default move constructor */
        BINEDepthwiseConvolutionLayer(BINEDepthwiseConvolutionLayer &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEDepthwiseConvolutionLayer &operator=(const BINEDepthwiseConvolutionLayer &) = delete;
        /** Default move assignment operator */
        BINEDepthwiseConvolutionLayer &operator=(BINEDepthwiseConvolutionLayer &&) = default;
        /** Default destructor */
        ~BINEDepthwiseConvolutionLayer();
        /** Initialize the function's source, destination, weights and convolution information.
         *
         * Valid data layouts:
         * - NHWC
         * - NCHW
         *
         * Valid data type configurations:
         * |src0           |src1               |src2   |dst            |
         * |:--------------|:------------------|:------|:--------------|
         * |F16            |F16                |F16    |F16            |
         * |F32            |F32                |F32    |F32            |
         * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
         * |QASYMM8        |QSYMM8_PER_CHANNEL |S32    |QASYMM8        |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
         * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32    |QASYMM8_SIGNED |
         *
         * @param[in, out] input            Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
         * @param[out]     output           Destination tensor. Data type supported: same as @p input.
         * @param[in]      weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
         *                                  Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
         * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                                  Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
         * @param[in]      conv_info        Padding and stride information to use for the convolution.
         * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
         * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation.
         * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         */
        void configure(BIITensor                   *input,
                       const BIITensor             *weights,
                       const BIITensor             *biases,
                       BIITensor                   *output,
                       const BIPadStrideInfo       &conv_info,
                       unsigned int                 depth_multiplier = 1,
                       const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                       const Size2D                &dilation         = Size2D(1U, 1U));

        /** Static function to check if given info will lead to a valid configuration of @ref NEDepthwiseConvolutionLayer
         *
         * @param[in] input            Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
         * @param[in] output           Destination tensor. Data type supported: same as @p input.
         * @param[in] weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
         *                             Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
         * @param[in] biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
         *                             Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
         * @param[in] conv_info        Padding and stride information to use for the convolution.
         * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
         * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
         * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo         *input,
                                 const BIITensorInfo         *weights,
                                 const BIITensorInfo         *biases,
                                 const BIITensorInfo         *output,
                                 const BIPadStrideInfo       &conv_info,
                                 unsigned int                 depth_multiplier = 1,
                                 const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                                 const Size2D                &dilation         = Size2D(1U, 1U));

        // Inherited methods overriden:
        void run() override;
        void prepare() override;

    private:
        /** Basic function to execute optimized depthwise convolution routines. This function calls the following kernels:
        *
        * @note At the moment 3x3 and 5x5 convolution of stride 1, 2 are supported
        *
        * -# NEFillBorderKernel (if pad_x or pad_y > 0) and no assembly kernel implementation is present
        * -# NEDepthwiseConvolutionLayer3x3Kernel if 3x3 and no assembly kernel implementation is present
        * -# cpu::CpuDepthwiseConvolutionAssemblyDispatch if assembly kernel implementation is present
        * -# NEDirectConvolutionLayerOutputStageKernel if re-quantization of output is required
        * -# @ref NEActivationLayer if fused activation is required
        *
        */
        class BINEDepthwiseConvolutionLayerOptimizedInternal : public BIIFunction
        {
        public:
            /** Default constructor */
            BINEDepthwiseConvolutionLayerOptimizedInternal(std::shared_ptr<BIIMemoryManager> memory_manager = nullptr);
            /** Prevent instances of this class from being copied (As this class contains pointers) */
            BINEDepthwiseConvolutionLayerOptimizedInternal(const BINEDepthwiseConvolutionLayerOptimizedInternal &) = delete;
            /** Default move constructor */
            BINEDepthwiseConvolutionLayerOptimizedInternal(BINEDepthwiseConvolutionLayerOptimizedInternal &&) = default;
            /** Prevent instances of this class from being copied (As this class contains pointers) */
            BINEDepthwiseConvolutionLayerOptimizedInternal &
            operator=(const BINEDepthwiseConvolutionLayerOptimizedInternal &) = delete;
            /** Default move assignment operator */
            BINEDepthwiseConvolutionLayerOptimizedInternal &
            operator=(BINEDepthwiseConvolutionLayerOptimizedInternal &&) = default;
            /** Default destructor */
            ~BINEDepthwiseConvolutionLayerOptimizedInternal() = default;
            /** Initialize the function's source, destination, kernels and border_size.
             *
             * @param[in, out] input            Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
             * @param[in]      weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input.
             * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
             *                                  Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
             * @param[out]     output           Destination tensor. Data type supported: same as @p input.
             * @param[in]      conv_info        Padding and stride information to use for the convolution.
             * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
             * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation.
             * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
             */
            void configure(BIITensor                   *input,
                           const BIITensor             *weights,
                           const BIITensor             *biases,
                           BIITensor                   *output,
                           const BIPadStrideInfo       &conv_info,
                           unsigned int                 depth_multiplier = 1,
                           const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                           const Size2D                &dilation         = Size2D(1U, 1U));

            /** Static function to check if given info will lead to a valid configuration of @ref NEDepthwiseConvolutionLayer3x3
             *
             * @param[in] input            Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
             * @param[in] weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input.
             * @param[in] biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
             *                             Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
             * @param[in] output           Destination tensor. Data type supported: same as @p input.
             * @param[in] conv_info        Padding and stride information to use for the convolution.
             * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
             * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
             * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo         *input,
                                     const BIITensorInfo         *weights,
                                     const BIITensorInfo         *biases,
                                     const BIITensorInfo         *output,
                                     const BIPadStrideInfo       &conv_info,
                                     unsigned int                 depth_multiplier = 1,
                                     const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                                     const Size2D                &dilation         = Size2D(1U, 1U));

            // Inherited methods overriden:
            void run() override;
            void prepare() override;

        private:
            BIMemoryGroup _memory_group;
            struct Impl;
            std::unique_ptr<Impl> _impl;
        };

        /** Basic function to execute a generic depthwise convolution. This function calls the following kernel:
         *
         * -# @ref NEDepthwiseConvolutionLayerNativeKernel
         *
         */
        class BINEDepthwiseConvolutionLayerGeneric : public BIIFunction
        {
        public:
            /** Default constructor */
            BINEDepthwiseConvolutionLayerGeneric();
            /** Prevent instances of this class from being copied (As this class contains pointers) */
            BINEDepthwiseConvolutionLayerGeneric(const BINEDepthwiseConvolutionLayerGeneric &) = delete;
            /** Default move constructor */
            BINEDepthwiseConvolutionLayerGeneric(BINEDepthwiseConvolutionLayerGeneric &&) = default;
            /** Prevent instances of this class from being copied (As this class contains pointers) */
            BINEDepthwiseConvolutionLayerGeneric &operator=(const BINEDepthwiseConvolutionLayerGeneric &) = delete;
            /** Default move assignment operator */
            BINEDepthwiseConvolutionLayerGeneric &operator=(BINEDepthwiseConvolutionLayerGeneric &&) = default;
            /** Default destructor */
            ~BINEDepthwiseConvolutionLayerGeneric() = default;
            /** Initialize the function's source, destination, weights and convolution information.
             *
             * @param[in, out] input            Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
             * @param[out]     output           Destination tensor. Data type supported: same as @p input.
             * @param[in]      weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
             *                                  Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
             * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
             *                                  Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
             * @param[in]      conv_info        Padding and stride information to use for the convolution.
             * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
             * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation.
             * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
             */
            void configure(BIITensor                   *input,
                           const BIITensor             *weights,
                           const BIITensor             *biases,
                           BIITensor                   *output,
                           const BIPadStrideInfo       &conv_info,
                           unsigned int                 depth_multiplier = 1,
                           const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                           const Size2D                &dilation         = Size2D(1U, 1U));

            /** Static function to check if given info will lead to a valid configuration of @ref NEDepthwiseConvolutionLayerGeneric
             *
             * @param[in] input            Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32. (Written to only for border filling).
             * @param[in] output           Destination tensor. Data type supported: same as @p input.
             * @param[in] weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
             *                             Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
             * @param[in] biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
             *                             Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
             * @param[in] conv_info        Padding and stride information to use for the convolution.
             * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
             * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
             * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo         *input,
                                     const BIITensorInfo         *weights,
                                     const BIITensorInfo         *biases,
                                     const BIITensorInfo         *output,
                                     const BIPadStrideInfo       &conv_info,
                                     unsigned int                depth_multiplier = 1,
                                     const BIActivationLayerInfo &act_info         = BIActivationLayerInfo(),
                                     const Size2D                &dilation         = Size2D(1U, 1U));

            // Inherited methods overriden:
            void run() override;

        private:
            struct Impl;
            std::unique_ptr<Impl> _impl;
        };
        BIMemoryGroup _memory_group;
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
