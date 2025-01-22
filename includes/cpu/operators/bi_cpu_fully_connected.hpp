//
// Created by Mason on 2025/1/22.
//

#pragma once

#include <data/core/bi_tensor_info.hpp>
#include <function_info/bi_fullyConnectedLayerInfo.h>

#include <cpu/bi_i_cpu_operator.hpp>
#include <cpu/kernels/bi_cpu_transpose_kernel.hpp>

#include <memory>

namespace BatmanInfer {
    namespace cpu {
        // 前向声明
        class BICpuConvertFullyConnectedWeights;

        class BICpuFlatten;

        class BICpuGemm;

        class BICpuGemmLowpMatrixMultiplyCore;

        class BICpuFullyConnected : public BIICpuOperator {
        public:
            BICpuFullyConnected();

            ~BICpuFullyConnected();

            /**
             *
             * Valid data layouts:
             * - NHWC
             *
             * Valid data type configurations:
             * |src0           |src1               |src2   |dst            |
             * |:--------------|:------------------|:------|:--------------|
             * |F16            |F16                |F16    |F16            |
             * |F32            |F32                |F32    |F32            |
             * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
             * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
             * @param src Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
             * @param weights Weights tensor info. The weights must be 2 dimensional
             *                If this function is called after a Convolution Layer, the (transposed) weights will have as many rows as the product of the first 3 input's dimensions.
             *                If it is called after another FullyConnected Layer, the (transposed) weights will have as many rows as the input's first dimension.
             *                Data type supported: Same as @p src..
             * @param biases Bias tensor info. Can be nullptr. Data type supported: Same as @p weights, S32 if @p weights is QASYMM8/QASYMM8_SIGNED.
             * @param dst
             * @param fc_info (Optional) Fully connected layer additional info
             * @param weights_info (Optional) Stores necessary compute information when weights are already reshaped
             */
            void configure(const BIITensorInfo *src,
                           const BIITensorInfo *weights,
                           const BIITensorInfo *biases,
                           BIITensorInfo *dst,
                           BIFullyConnectedLayerInfo fc_info = BIFullyConnectedLayerInfo(),
                           const BIWeightsInfo &weights_info = BIWeightsInfo());

            /** Static function to check if given info will lead to a valid configuration of @ref CpuFullyConnected
           *
           * Similar to @ref CpuFullyConnected::configure()
           *
           * @return a status
           */
            static BIStatus validate(const BIITensorInfo *src,
                                     const BIITensorInfo *weights,
                                     const BIITensorInfo *biases,
                                     const BIITensorInfo *dst,
                                     const BIFullyConnectedLayerInfo &fc_info = BIFullyConnectedLayerInfo(),
                                     const BIWeightsInfo &weights_info = BIWeightsInfo());

            /** Static function that queries whether there exists fixed-format kernel and if it exists it will return in the first argument in what format
             * weights are expected to be reshaped as defined by WeightFormat class. Apart from the first argument the rest of the arguments are the same
             * as in @ref CpuFullyConnectedLayer::validate() except that all arguments are required.
             *
             * @return a status
             */
            static BIStatus has_opt_impl(BatmanInfer::BIWeightFormat &expected_weight_format,
                                         const BIITensorInfo *src,
                                         const BIITensorInfo *weights,
                                         const BIITensorInfo *biases,
                                         const BIITensorInfo *dst,
                                         const BIFullyConnectedLayerInfo &fc_info,
                                         BIWeightsInfo weights_info);

            void run(BatmanInfer::BIITensorPack &tensors) override;

            void prepare(BatmanInfer::BIITensorPack &constants) override;

            experimental::BIMemoryRequirements workspace() const override;

        private:
            void configure_fc_fc(const BIITensorInfo *src,
                                 const BIITensorInfo *weights,
                                 const BIITensorInfo *biases,
                                 BIITensorInfo *dst,
                                 const BIActivationLayerInfo &act);

            void configure_conv_fc(const BIITensorInfo *src,
                                   const BIITensorInfo *weights,
                                   const BIITensorInfo *biases,
                                   BIITensorInfo *dst,
                                   const BIActivationLayerInfo &act);

            void configure_mm(const BIITensorInfo *src,
                              const BIITensorInfo *weights,
                              const BIITensorInfo *biases,
                              BIITensorInfo *dst,
                              const BIActivationLayerInfo &act);

            enum AuxTensorIdx {
                AsmGemmWorkspace = 0,
                Pretranspose,
                GemmTemp1,
                GemmTemp2,
                GemmTemp3,
                GemmTemp4,
                GemmTemp5,
                GemmTemp6,
                GemmTemp7,
                GemmTemp8,
                // Slots above (0-9) reserved for either CpuGemm or CpuGemmLowpMatrixMultiplyCore
                TransposedWeights, // 转置权重
                ConvertedWeights, // 转换权重
                FlattenedSrc,
                Count
            };

            std::unique_ptr<BICpuFlatten> _flatten;
            std::unique_ptr<BICpuConvertFullyConnectedWeights> _convert_weights;
            std::unique_ptr<kernels::BICpuTransposeKernel> _transpose_weights;
            std::unique_ptr<BICpuGemm> _mm_gemm;
            std::unique_ptr<BICpuGemmLowpMatrixMultiplyCore> _mm_gemmlowp;

            BITensorInfo _flattened_src;
            BITensorInfo _converted_weights;
            BITensorInfo _reshaped_weights;
            BITensorInfo _trans_weights;
            AuxTensorIdx _trans_weights_idx;

            experimental::BIMemoryRequirements _aux_mem;

            bool _needs_weights_conversion;
            bool _needs_weights_reshape;
            bool _is_fc_after_conv;
            bool _is_quantized_asymmetric;
            bool _is_prepared;
            bool _enable_fast_math;
            bool _fixed_format;
            BatmanInfer::BIWeightFormat _weight_format;
            bool _dynamic_weights;

#ifdef BI_COMPUTE_ASSERTS_ENABLED
            int _asrt_run_count{};
            int _asrt_prepare_count{};
#endif // BI_COMPUTE_ASSERTS_ENABLED

        };
    }
}