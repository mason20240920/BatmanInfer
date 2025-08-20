//
// Created by Mason on 2025/1/17.
//

#pragma once

#include <data/core/bi_rounding.h>
#include <data/core/bi_types.hpp>
#include <function_info/bi_activationLayerInfo.h>
#include <runtime/bi_i_function.hpp>

#include <memory>

namespace BatmanInfer {
    class BIITensor;

    class BIITensorInfo;

    class BINEPixelWiseMultiplication : public BIIFunction {
    public:
        BINEPixelWiseMultiplication();

        ~BINEPixelWiseMultiplication();

        BINEPixelWiseMultiplication(const BINEPixelWiseMultiplication &) = delete;

        BINEPixelWiseMultiplication(BINEPixelWiseMultiplication &&) = default;

        BINEPixelWiseMultiplication &operator=(const BINEPixelWiseMultiplication &) = delete;

        BINEPixelWiseMultiplication &operator=(BINEPixelWiseMultiplication &&) = default;

        /** Initialise the kernel's inputs, output and convertion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |QSYMM16        |QSYMM16        |QASYMM16       |
     * |QSYMM16        |QSYMM16        |S32            |
     * |U8             |U8             |U8             |
     * |U8             |U8             |S16            |
     * |U8             |S16            |S16            |
     * |S16            |U8             |S16            |
     * |S16            |S16            |S16            |
     * |S32            |S32            |S32            |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *
     * @param[in, out] input1          An input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     *                                 This input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          An input tensor. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), QASYMM8_SIGNED (only if @p input1 is QASYMM8_SIGNED), S16, S32, QSYMM16 (only if @p input1 is QSYMM16), F16 (only if @p input1 is F16), F32 (only if @p input1 is F32).
     *                                 This input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported:
     *                                 - U8, only if both inputs are U8.
     *                                 - QASYMM8, only if both inputs are QASYMM8.
     *                                 - QASYMM8_SIGNED, only if @p input1 is QASYMM8_SIGNED.
     *                                 - S16.
     *                                 - QSYMM16, only if both inputs are QSYMM16.
     *                                 - S32, only if both inputs are S32 or both are QSYMM16.
     *                                 - F16, only if @p input1 is F16.
     *                                 - F32, only if both inputs are F32.
     * @param[in]      scale           Scale to apply after multiplication.
     *                                 Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     *                                 If both @p input1, @p input2 and @p output are of datatype S32, scale cannot be 1/255
     * @param[in]      overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the inputs is of quantized datatype
     * @param[in]      rounding_policy Rounding policy.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
        void configure(const BIITensor *input1,
                       const BIITensor *input2,
                       BIITensor *output,
                       float scale,
                       BIConvertPolicy overflow_policy,
                       BIRoundingPolicy rounding_policy,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        void dynamic_configure() const;

        /** Static function to check if given info will lead to a valid configuration of @ref NEPixelWiseMultiplication
         *
         * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
         *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
         *
         * @param[in] input1          An input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
         * @param[in] input2          An input tensor info. Data types supported: U8, QASYMM8 (only if @p input1 is QASYMM8), QASYMM8_SIGNED (only if @p input1 is QASYMM8_SIGNED), S16, S32, QSYMM16 (only if both inputs are QSYMM16), F16 (only if @p input1 is F16), F32 (only if @p input1 is F32).
         * @param[in] output          Output tensor info. Data types supported:
         *                            - U8, only if both inputs are U8.
         *                            - QASYMM8, only if both inputs are QASYMM8.
         *                            - QASYMM8_SIGNED, only if @p input1 is QASYMM8_SIGNED.
         *                            - S16.
         *                            - QSYMM16, only if both inputs are QSYMM16.
         *                            - S32, only if both inputs are S32 or both are QSYMM16.
         *                            - F16, only if @p input1 is F16.
         *                            - F32, only if both inputs are F32.
         * @param[in] scale           Scale to apply after multiplication.
         *                            Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
         *                            If both @p input1, @p input2 and @p output are of datatype S32, scale cannot be 1/255
         * @param[in] overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the inputs is of quantized datatype
         * @param[in] rounding_policy Rounding policy.
         * @param[in] act_info        (Optional) Activation layer information in case of a fused activation. Currently not supported.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input1,
                                 const BIITensorInfo *input2,
                                 const BIITensorInfo *output,
                                 float scale,
                                 BIConvertPolicy overflow_policy,
                                 BIRoundingPolicy rounding_policy,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        void run();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

    /** Basic function to run cpu::CpuComplexMul. */
    class BINEComplexPixelWiseMultiplication : public BIIFunction {
    public:
        /** Default Constructor */
        BINEComplexPixelWiseMultiplication();

        /** Default Destructor */
        ~BINEComplexPixelWiseMultiplication();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEComplexPixelWiseMultiplication(const BINEComplexPixelWiseMultiplication &) = delete;

        /** Default move constructor */
        BINEComplexPixelWiseMultiplication(BINEComplexPixelWiseMultiplication &&) = default;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEComplexPixelWiseMultiplication &operator=(const BINEComplexPixelWiseMultiplication &) = delete;

        /** Default move assignment operator */
        BINEComplexPixelWiseMultiplication &operator=(BINEComplexPixelWiseMultiplication &&) = default;

        /** Initialise the kernel's inputs, output.
         *
         * @param[in, out] input1   An input tensor. Data types supported: F32. Number of channels supported: 2 (complex tensor).
         *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
         * @param[in, out] input2   An input tensor. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
         *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
         * @param[out]     output   The output tensor. Data types supported: same as @p input1. Number of channels: same as @p input1.
         * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         */
        void configure(BIITensor *input1,
                       BIITensor *input2,
                       BIITensor *output,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        /** Static function to check if given info will lead to a valid configuration of @ref NEComplexPixelWiseMultiplication
         *
         * @param[in] input1   An input tensor info. Data types supported: F32. Number of channels supported: 2 (complex tensor).
         * @param[in] input2   An input tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
         * @param[in] output   The output tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
         * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         */
        static BIStatus validate(const BIITensorInfo *input1,
                                 const BIITensorInfo *input2,
                                 const BIITensorInfo *output,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        // Inherited methods overridden:
        void run();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
}