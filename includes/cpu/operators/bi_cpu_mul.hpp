//
// Created by Mason on 2025/1/17.
//

#pragma once

#include <data/core/bi_i_tensor_info.hpp>
#include <function_info/bi_activationLayerInfo.h>

#include <cpu/bi_i_cpu_operator.hpp>

namespace BatmanInfer {
    namespace cpu {
        /**
         * Basic function to run @ref kernels::CpuMulKernel
         */
        class BICpuMul : public BIICpuOperator {
        public:
            /** Initialise the kernel's inputs, dst and convert policy.
             *
             * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
             *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
             *
             * @param[in, out] src1            First input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
             *                                 This input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
             * @param[in, out] src2            Second input tensor info. Data types supported: U8, QASYMM8 (only if @p src1 is QASYMM8), QASYMM8_SIGNED (only if @p src1 is QASYMM8_SIGNED), S16, S32, QSYMM16 (only if @p src1 is QSYMM16), F16 (only if @p src1 is F16), F32 (only if @p src1 is F32).
             *                                 This input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
             * @param[out]     dst             dst tensor info. Data types supported:
             *                                 - U8, only if both inputs are U8.
             *                                 - QASYMM8, only if both inputs are QASYMM8.
             *                                 - QASYMM8_SIGNED, only if @p src1 is QASYMM8_SIGNED.
             *                                 - S16.
             *                                 - QSYMM16, only if both inputs are QSYMM16.
             *                                 - S32, only if both inputs are S32 or both are QSYMM16.
             *                                 - F16, only if @p src1 is F16.
             *                                 - F32, only if both inputs are F32.
             * @param[in]      scale           Scale to apply after multiplication.
             *                                 Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
             *                                 If both @p src1, @p src2 and @p dst are of datatype S32, scale cannot be 1/255
             * @param[in]      overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the inputs is of quantized datatype
             * @param[in]      rounding_policy Rounding policy.
             * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation. Currently not supported.
             */
            void configure(BIITensorInfo *src1,
                           BIITensorInfo *src2,
                           BIITensorInfo *dst,
                           float scale,
                           BIConvertPolicy overflow_policy,
                           BIRoundingPolicy rounding_policy,
                           const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref CpuMul::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *src1,
                                     const BIITensorInfo *src2,
                                     const BIITensorInfo *dst,
                                     float scale,
                                     BIConvertPolicy overflow_policy,
                                     BIRoundingPolicy rounding_policy,
                                     const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

            // Inherited methods overridden:
            void run(BIITensorPack &tensors) override;
        };

        /** Basic function to run @ref kernels::CpuComplexMulKernel */
        class BICpuComplexMul : public BIICpuOperator {
        public:
            /** Initialise the kernel's inputs, dst.
             *
             * @param[in, out] src1     First input tensor. Data types supported: F32. Number of channels supported: 2 (complex tensor).
             *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
             * @param[in, out] src2     Second input tensor. Data types supported: same as @p src1. Number of channels supported: same as @p src1.
             *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
             * @param[out]     dst      The dst tensor. Data types supported: same as @p src1. Number of channels: same as @p src1.
             * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
             */
            void configure(BIITensorInfo *src1,
                           BIITensorInfo *src2,
                           BIITensorInfo *dst,
                           const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref CpuComplexMul::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *src1,
                                     const BIITensorInfo *src2,
                                     const BIITensorInfo *dst,
                                     const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

            // Inherited methods overridden:
            void run(BIITensorPack &tensors) override;
        };
    } //namespace cpu
}
