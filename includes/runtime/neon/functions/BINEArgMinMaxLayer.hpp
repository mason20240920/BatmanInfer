//
// Created by Mason on 2025/4/10.
//

#pragma once
/** @file
 * @publicapi
 */

#include <data/core/bi_types.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/neon/bi_i_ne_simple_function.h>

namespace BatmanInfer {
    class BIITensor;
    /** Function to calculate the index of the minimum or maximum values in a
     *  tensor based on an axis.
     *
     *  This function calls the following kernels:
     *
     * -# NEReductionOperationKernel
     * -# NEFillBorderKernel
     *
     * @note The default data type for an uninitialized output tensor is
     *       signed 32-bit integer (S32). It is the user's responsibility to check
     *       that the results do not overflow because the indices are computed
     *       in unsigned 32-bit (U32).
     */
    class BINEArgMinMaxLayer : public BIIFunction {
    public:
        /** Constructor */
        BINEArgMinMaxLayer(std::shared_ptr<BIIMemoryManager> memory_manager = nullptr);

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEArgMinMaxLayer(const BINEArgMinMaxLayer &) = delete;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEArgMinMaxLayer &operator=(const BINEArgMinMaxLayer &) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEArgMinMaxLayer(BINEArgMinMaxLayer &&) = delete;

        /** Prevent instances of this class from being moved (As this class contains non movable objects) */
        BINEArgMinMaxLayer &operator=(BINEArgMinMaxLayer &&) = delete;

        /** Default destructor */
        ~BINEArgMinMaxLayer();

        /** Set the input and output tensors.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src            |dst           |
         * |:--------------|:-------------|
         * |QASYMM8        |U32, S32      |
         * |QASYMM8_SIGNED |U32, S32      |
         * |S32            |U32, S32, S64 |
         * |F16            |U32, S32      |
         * |F32            |U32, S32      |
         *
         * @param[in]  input  Input source tensor. Data types supported: QASYMM8_SIGNED/QASYMM8/S32/F16/F32.
         * @param[in]  axis   Axis to find max/min index.
         * @param[out] output Output source tensor. Data types supported: U32/S32.
         * @param[in]  op     Operation to perform: min or max
         */
        void configure(BIITensor *input, int axis, BIITensor *output, const BIReductionOperation &op);

        /** Static function to check if given info will lead to a valid configuration of @ref NEArgMinMaxLayer
         *
         * @param[in] input  Input source tensor info. Data types supported: QASYMM8_SIGNED/QASYMM8/S32/F16/F32.
         * @param[in] axis   Axis to find max/min index.
         * @param[in] output Output source tensor info. Data types supported: U32/S32/S64.
         * @param[in] op     Operation to perform: min or max
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *input, int axis, const BIITensorInfo *output,
                                 const BIReductionOperation &op);

        // Inherited methods overridden:
        void run();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace BatmanInfer
