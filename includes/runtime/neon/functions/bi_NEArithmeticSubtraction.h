//
// Created by holynova on 2025/1/17.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "function_info/bi_activationLayerInfo.h"
#include "runtime/bi_i_function.hpp"
#include "runtime/neon/bi_i_ne_operator.hpp"

namespace BatmanInfer {

    class BIITensor;

    /** Basic function to run cpu::kernels::CpuSubKernel
     *
     * @note The tensor data type for the inputs must be U8/QASYMM8/S16/S32/F16/F32.
     * @note The function performs an arithmetic subtraction between two tensors.
     *
     *  This function calls the following kernels:
     * -# cpu::kernels::CpuSubKernel
     */
    class BINEArithmeticSubtraction : public BIIFunction
    {
    public:
        /** Default Constructor */
        BINEArithmeticSubtraction();
        /** Default Destructor */
        ~BINEArithmeticSubtraction();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEArithmeticSubtraction(const BINEArithmeticSubtraction &) = delete;
        /** Default move constructor */
        BINEArithmeticSubtraction(BINEArithmeticSubtraction &&);
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEArithmeticSubtraction &operator=(const BINEArithmeticSubtraction &) = delete;
        /** Default move assignment operator */
        BINEArithmeticSubtraction &operator=(BINEArithmeticSubtraction &&);
        /** Initialise the kernel's inputs, output and conversion policy.
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
         * |U8             |U8             |U8             |
         * |S16            |S16            |S16            |
         * |S32            |S32            |S32            |
         * |F16            |F16            |F16            |
         * |F32            |F32            |F32            |
         *
         * @param[in]  input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
         * @param[in]  input2   Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
         * @param[out] output   Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
         * @param[in]  policy   Policy to use to handle overflow. Convert policy cannot be WRAP if datatype is quantized.
         * @param[in]  act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         */
        void configure(const BIITensor             *input1,
                       const BIITensor             *input2,
                       BIITensor                   *output,
                       BIConvertPolicy              policy,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());
        /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticSubtraction
         *
         * @param[in] input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/F16/F32
         * @param[in] input2   Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/F16/F32
         * @param[in] output   Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/F16/F32
         * @param[in] policy   Policy to use to handle overflow. Convert policy cannot be WRAP if datatype is quantized.
         * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo         *input1,
                                 const BIITensorInfo         *input2,
                                 const BIITensorInfo         *output,
                                 BIConvertPolicy              policy,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
