//
// Created by Mason on 2025/4/3.
//

#pragma once
#include <data/core/bi_types.hpp>
#include <runtime/neon/bi_i_ne_simple_function_no_border.h>

namespace BatmanInfer {
    class BIITensor;
    class BIITensorInfo;

    class BINESelect final : public BIINESimpleFunctionNoBorder {
    public:
        /** Initialise the kernel's inputs and output.
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src0           |src1           |src2   |dst            |
         * |:--------------|:--------------|:------|:--------------|
         * |U8             |All            |All    |All            |
         *
         * @param[in]  c      Condition input tensor. Data types supported: U8.
         * @param[in]  x      First input tensor. Data types supported: All.
         * @param[in]  y      Second input tensor. Data types supported: Same as @p x
         * @param[out] output Output tensor. Data types supported: Same as @p x.
         */
        void configure(const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output);

        /** Static function to check if given info will lead to a valid configuration of @ref NESelect
         *
         * @param[in] c      Condition input tensor. Data types supported: U8.
         * @param[in] x      First input tensor. Data types supported: All.
         * @param[in] y      Second input tensor. Data types supported: Same as @p x
         * @param[in] output Output tensor. Data types supported: Same as @p x.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *c, const BIITensorInfo *x, const BIITensorInfo *y,
                                 const BIITensorInfo *output);
    };
} // namespace arm_compute
