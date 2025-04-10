//
// Created by Mason on 2025/4/10.
//

#pragma once

#include <runtime/bi_i_function.hpp>
#include <runtime/neon/functions/bi_ne_reshape_layer.hpp>
#include <runtime/bi_tensor.hpp>

#include <memory>

namespace BatmanInfer {
    class BIITensor;
    class BINEReductionOperationKernel;

    /** Basic function to simulate a reduction operation. This function calls the following kernels:
     *
     * -# @ref BINEReshapeLayer
     * -# BINEReductionOperationKernel
     *
     */
    class BINEReductionOperation : public BIIFunction {
    public:
        /** Default constructor */
        BINEReductionOperation(std::shared_ptr<BIIMemoryManager> memory_manager = nullptr);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEReductionOperation(const BINEReductionOperation &) = delete;

        /** Default move constructor */
        BINEReductionOperation(BINEReductionOperation &&) = default;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEReductionOperation &operator=(const BINEReductionOperation &) = delete;

        /** Default move assignment operator */
        BINEReductionOperation &operator=(BINEReductionOperation &&) = default;

        /** Default destructor */
        ~BINEReductionOperation();

        /** Set the input and output tensors.
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
         * |S32            |S32            |
         *
         * @param[in, out] input     Source tensor. Data type supported: QASYMM8_SIGNED/QASYMM8/F16/F32/S32. (Written to only for border_size != 0)
         * @param[out]     output    Destination tensor. Data types and data layouts supported: same as @p input.
         * @param[in]      axis      Dimension along which to reduce. Supported reduction axis : 0
         * @param[in]      op        Reduction operation to perform.
         * @param[in]      keep_dims (Optional) Whether to keep the reduced dimension after the operation. Defaults to true.
         */
        void configure(BIITensor *input, BIITensor *output, unsigned int axis, BIReductionOperation op,
                       bool keep_dims = true);

        /** Static function to check if given info will lead to a valid configuration of @ref NEReductionOperation.
         *
         * @param[in] input     Source tensor info. Data type supported: QASYMM8_SIGNED/QASYMM8/F16/F32/S32.
         * @param[in] output    Destination tensor info. Data types and data layouts supported: same as @p input.
         * @param[in] axis      Dimension along which to reduce. Supported reduction axis : 0
         * @param[in] op        Reduction operation to perform.
         * @param[in] keep_dims (Optional) Whether to keep the reduced dimension after the operation. Defaults to true.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *output,
                                 unsigned int axis,
                                 BIReductionOperation op,
                                 bool keep_dims = true);

        // Inherited methods overridden:
        void run() override;

    private:
        BIMemoryGroup _memory_group;
        std::unique_ptr<BINEReductionOperationKernel> _reduction_kernel;
        BINEReshapeLayer _reshape;
        BITensor _output_internal;
        size_t _window_split;
        int _reduction_axis;
        bool _is_reshape_required;
    };
}
