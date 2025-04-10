//
// Created by Mason on 2025/4/10.
//

#pragma once

#include <data/core/neon/bi_i_ne_kernel.hpp>

namespace BatmanInfer {
    class BIITensor;

    /** Kernel to perform a reduction operation
     *
     * @note For ARG_MIN/ARG_MAX reduction, the default data type for an uninitialized
     *       output tensor is signed 32-bit integer (S32). It is the user's responsibility
     *       to check that the results do not overflow because the indices are computed
     *       in unsigned 32-bit (U32).
     */
    class BINEReductionOperationKernel : public BIINEKernel {
    public:
        const char *name() const override {
            return "BINEReductionOperationKernel";
        }

        /** Default constructor */
        BINEReductionOperationKernel();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEReductionOperationKernel(const BINEReductionOperationKernel &) = delete;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEReductionOperationKernel &operator=(const BINEReductionOperationKernel &) = delete;

        /** Allow instances of this class to be moved */
        BINEReductionOperationKernel(BINEReductionOperationKernel &&) = default;

        /** Allow instances of this class to be moved */
        BINEReductionOperationKernel &operator=(BINEReductionOperationKernel &&) = default;

        /** Default destructor */
        ~BINEReductionOperationKernel() = default;

        /** Set the source, destination of the kernel
         *
         * @param[in]  input  Source tensor. Data type supported: QASYMM8_SIGNED/QASYMM8/F16/F32/S32.
         * @param[out] output Destination tensor.Data types and data layouts supported: same as @p input, S32 for ARG_MIX/ARG_MAX.
         *                    Output will have the same number of dimensions as input.
         * @param[in]  axis   Axis along which to reduce. Supported reduction axis : 0
         * @param[in]  op     Reduction operation to perform.
         */
        void configure(const BIITensor *input, BIITensor *output, unsigned int axis, BIReductionOperation op);

        /** Static function to check if given info will lead to a valid configuration of @ref NEReductionOperationKernel.
         *
         * @param[in] input  Source tensor info. Data type supported: QASYMM8_SIGNED/QASYMM8/F16/F32/S32.
         * @param[in] output Destination tensor info.Data types and data layouts supported: same as @p input, S32 for ARG_MIX/ARG_MAX.
         *                   Output will have the same number of dimensions as input.
         * @param[in] axis   Axis along which to reduce. Supported reduction axis : 0
         * @param[in] op     Reduction operation to perform.
         *
         * @return a status
         */
        static BIStatus
        validate(const BIITensorInfo *input, const BIITensorInfo *output, unsigned int axis, BIReductionOperation op);

    private:
        // Inherited methods overridden:
        void run(const BIWindow &window, const ThreadInfo &info) override;

        /** Common signature for all the specialized Reduction functions
         *
         * @param[in] window Region on which to execute the kernel.
         */
        using ReductionFunction = void (*)(const BIWindow &window, const BIITensor *in, BIITensor *out,
                                           BIReductionOperation op);

        /** Populate the _func with the right reduction operation handler
        */
        void reduce_op();

        ReductionFunction _func;
        const BIITensor *_input;
        BIITensor *_output;
        unsigned int _reduction_axis;
        BIReductionOperation _op;
    };
} // namespace arm_compute
