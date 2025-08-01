//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "function_info/bi_activationLayerInfo.h"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_runtime_context.hpp"

#include <memory>

namespace BatmanInfer {
    // Forward declarations
    class BIITensor;
    class BIITensorInfo;

    /** Basic function to run cpu::kernels::CpuActivationKernel
     *
     * @note The function simulates an activation layer with the specified activation function.
     */
    class BINEActivationLayer : public BIIFunction {
    public:
        /** Constructor
         *
         * @param[in] ctx Runtime context to be used by the function
         */
        BINEActivationLayer(BIIRuntimeContext *ctx = nullptr);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEActivationLayer(const BINEActivationLayer &) = delete;

        /** Default move constructor */
        BINEActivationLayer(BINEActivationLayer &&);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEActivationLayer &operator=(const BINEActivationLayer &) = delete;

        /** Default move assignment operator */
        BINEActivationLayer &operator=(BINEActivationLayer &&);

        /** Destructor */
        ~BINEActivationLayer();

        /** [NEActivationLayer snippet] **/
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
         * |QSYMM16        |QSYMM16        |
         * |F16            |F16            |
         * |F32            |F32            |
         *
         * @note If the output tensor is a nullptr or is equal to the input, the activation function will be performed in-place
         *
         * @param[in, out] input           Source tensor. In case of @p output tensor = nullptr, this tensor will store the result
         *                                 of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
         * @param[out]     output          Destination tensor. Data type supported: same as @p input
         * @param[in]      activation_info Activation layer parameters.
         */
        void configure(BIITensor *input, BIITensor *output, BIActivationLayerInfo activation_info);

        void dynamic_configure(BIITensor *input);

        /** [NEActivationLayer snippet] **/
        /** Static function to check if given info will lead to a valid configuration of @ref NEActivationLayer
         *
         * @param[in] input    Source tensor info. In case of @p output tensor info = nullptr, this tensor will store the result
         *                     of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
         * @param[in] output   Destination tensor info. Data type supported: same as @p input
         * @param[in] act_info Activation layer information.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *output,
                                 const BIActivationLayerInfo &act_info);

        // Inherited methods overridden
        void run();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace BatmanInfer
