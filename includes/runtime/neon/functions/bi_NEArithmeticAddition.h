//
// Created by holynova on 2025/1/17.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "function_info/bi_activationLayerInfo.h"
#include "runtime/bi_i_function.hpp"

#include <memory>

namespace BatmanInfer {
    class BIITensor;
    class BIITensorInfo;

    /** Basic function to run cpu::kernels::CpuAddKernel */
    class BINEArithmeticAddition : public BIIFunction {
    public:
        /** Default Constructor */
        BINEArithmeticAddition();

        /** Default Destructor */
        ~BINEArithmeticAddition();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEArithmeticAddition(const BINEArithmeticAddition &) = delete;

        /** Default move constructor */
        BINEArithmeticAddition(BINEArithmeticAddition &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEArithmeticAddition &operator=(const BINEArithmeticAddition &) = delete;

        /** Default move assignment operator */
        BINEArithmeticAddition &operator=(BINEArithmeticAddition &&);

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
         * |QSYMM16        |QSYMM16        |S32            |
         * |U8             |U8             |U8             |
         * |S16            |S16            |S16            |
         * |S32            |S32            |S32            |
         * |F16            |F16            |F16            |
         * |F32            |F32            |F32            |
         *
         * @param[in]  input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
         * @param[in]  input2   Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
         * @param[out] output   Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
         * @param[in]  policy   Policy to use to handle overflow.
         * @param[in]  act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         */
        void configure(const BIITensor *input1,
                       const BIITensor *input2,
                       BIITensor *output,
                       BIConvertPolicy policy,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        /**
         * @brief Initialise the kernel's inputs, output and conversion policy.
         * @param input1
         * @param input2
         * @param is_tile_matrix 是否是Mask相加矩阵
         */
        void dynamic_configure(const BIITensor *input1, const BIITensor *input2, bool is_tile_matrix);

        /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticAddition
         *
         * @param[in] input1   First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
         * @param[in] input2   Second tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
         * @param[in] output   Output tensor info. Data types supported: U8/SQASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
         * @param[in] policy   Policy to use to handle overflow
         * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input1,
                                 const BIITensorInfo *input2,
                                 const BIITensorInfo *output,
                                 BIConvertPolicy policy,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace BatmanInfer
