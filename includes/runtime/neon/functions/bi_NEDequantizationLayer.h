//
// Created by holynova on 2025/1/17.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/bi_i_function.hpp"

#include <memory>

namespace BatmanInfer {
    // Forward declarations
    class BIITensor;
    class BIITensorInfo;

    /** Basic function to run cpu::CpuDequantize that dequantizes an input tensor */
    class BINEDequantizationLayer : public BIIFunction {
    public:
        /** Default Constructor */
        BINEDequantizationLayer();

        /** Default Destructor */
        ~BINEDequantizationLayer();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEDequantizationLayer(const BINEDequantizationLayer &) = delete;

        /** Default move constructor */
        BINEDequantizationLayer(BINEDequantizationLayer &&) = default;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEDequantizationLayer &operator=(const BINEDequantizationLayer &) = delete;

        /** Default move assignment operator */
        BINEDequantizationLayer &operator=(BINEDequantizationLayer &&) = default;

        /** Configure the kernel.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src                |dst         |
         * |:------------------|:-----------|
         * |QASYMM8            |F16, F32    |
         * |QASYMM8_SIGNED     |F16, F32    |
         * |QSYMM8_PER_CHANNEL |F16, F32    |
         * |QSYMM8             |F16, F32    |
         * |QSYMM16            |F16, F32    |
         *
         * @param[in]  input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/QSYMM8/QSYMM16.
         * @param[out] output Destination tensor with the same dimensions of input. Data type supported: F16/F32.
         */
        void configure(const BIITensor *input, BIITensor *output);

        void dynamic_configure(const BIITensor *input) const;

        /** Static function to check if given info will lead to a valid configuration of @ref NEDequantizationLayer
         *
         * @param[in] input  Input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/QSYMM8/QSYMM16.
         * @param[in] output Output tensor info. Data type supported: F16/F32.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *output);

        // Inherited methods overridden:
        void run();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace BatmanInfer
