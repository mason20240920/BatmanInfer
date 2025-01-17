//
// Created by holynova on 2025/1/17.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_runtime_context.hpp"

#include <memory>

namespace BatmanInfer {

    class BIITensor;
    class BIITensorInfo;

    /** Basic function to run a quantization layer using cpu::CpuQuantize */
    class BINEQuantizationLayer : public BIIFunction
    {
    public:
        BINEQuantizationLayer();
        /** Default Destructor */
        ~BINEQuantizationLayer();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEQuantizationLayer(const BINEQuantizationLayer &) = delete;
        /** Default move constructor */
        BINEQuantizationLayer(BINEQuantizationLayer &&) = default;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEQuantizationLayer &operator=(const BINEQuantizationLayer &) = delete;
        /** Default move assignment operator */
        BINEQuantizationLayer &operator=(BINEQuantizationLayer &&) = default;
        /** Set the input and output tensors.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src                |dst                                    |
         * |:------------------|:--------------------------------------|
         * |QASYMM8            |QASYMM8, QASYMM8_SIGNED, QASYMM16      |
         * |QASYMM8_SIGNED     |QASYMM8, QASYMM8_SIGNED, QASYMM16      |
         * |F16                |QASYMM8, QASYMM8_SIGNED, QASYMM16      |
         * |F32                |QASYMM8, QASYMM8_SIGNED, QASYMM16      |
         *
         * @param[in]  input  Source tensor. The dimensions over the third will be interpreted as batches. Data types supported: QASYMM8/QASYMM8_SIGNED/F32/F16.
         * @param[out] output Destination tensor with the same dimensions of input. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16
         */
        void configure(const BIITensor *input, BIITensor *output);
        /** Static function to check if given info will lead to a valid configuration of @ref NEQuantizationLayer
         *
         * @param[in] input  Input tensor info. The dimensions over the third will be interpreted as batches. Data types supported: QASYMM8/QASYMM8_SIGNED/F32/F16.
         * @param[in] output Output tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *output);

        // Inherited methods overridden:
        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
