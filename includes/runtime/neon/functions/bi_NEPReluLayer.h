//
// Created by holynova on 2025/1/17.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/bi_i_function.hpp"

#include <memory>

namespace BatmanInfer {

    class BIITensor;
    class BIITensorInfo;

    /** Basic function to run cpu::kernels::CpuArithmeticKernel for PRELU
     *
     * @note The function implements an activation layer with the PRELU activation function.
     */
    class BINEPReluLayer : public BIIFunction
    {
    public:
        /** Default Constructor */
        BINEPReluLayer();
        /** Default Destructor */
        ~BINEPReluLayer();
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEPReluLayer(const BINEPReluLayer &) = delete;
        /** Default move constructor */
        BINEPReluLayer(BINEPReluLayer &&);
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEPReluLayer &operator=(const BINEPReluLayer &) = delete;
        /** Default move assignment operator */
        BINEPReluLayer &operator=(BINEPReluLayer &&);
        /** Set the input and output tensor.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src            |dst            |
         * |:--------------|:--------------|
         * |QASYMM8        |QASYMM8        |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED |
         * |F16            |F16            |
         * |F32            |F32            |
         *
         * @param[in]  input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
         * @param[in]  alpha  Source alpha tensor. Data types supported: same of @p input.
         * @param[out] output Destination tensor. Data type supported: same as @p input
         */
        void configure(const BIITensor *input, const BIITensor *alpha, BIITensor *output);
        /** Static function to check if given info will lead to a valid configuration of @ref NEPReluLayer
         *
         * @param[in] input  Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
         * @param[in] alpha  Source alpha tensor info. Data types supported: same of @p input.
         * @param[in] output Destination tensor info. Data type supported: same as @p input
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *alpha, const BIITensorInfo *output);

        // Inherited methods overridden:
        void run();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

} // namespace BatmanInfer
