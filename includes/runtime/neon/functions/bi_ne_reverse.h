//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/neon/bi_i_ne_simple_function_no_border.h"

namespace BatmanInfer {

    class BIITensor;
    class BIITensorInfo;

    /** Basic function to run NEReverseKernel */
    class BINEReverse : public BIINESimpleFunctionNoBorder
    {
    public:
        /** Initialize the function
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src0                        |src1           |dst                         |
         * |:---------------------------|:--------------|:---------------------------|
         * |All except SIZET <= 32-bits |U32, S32       |All except SIZET <= 32-bits |
         *
         * @param[in]  input             Input tensor. Data types supported: All except SIZET <= 32-bit data types
         * @param[out] output            Output tensor. Data type supported: Same as @p input
         * @param[in]  axis              Axis tensor. Contains the indices of the dimensions to reverse. Data type supported: U32/S32
         * @param[in]  use_inverted_axis Reverse ACL axis indices convention, if true, (inverted)axis = (tensor_rank - 1) - axis
         *
         * @note The value of each axis should be between [-rank, rank)
         * @note If there are duplicate values in the tensor, the subsequent axis values are ignored. e.g. an array of [2, 2] has the same effects as [2].
         *
         * @deprecated Support for U32 in axis tensor will be removed in 24.02 release
         *
         */
        void configure(const BIITensor *input, BIITensor *output, const BIITensor *axis, const bool use_inverted_axis = false);
        /** Static function to check if given info will lead to a valid configuration of NEReverseKernel
         *
         * Similar to @ref NEReverse::configure()
         */
        static BIStatus validate(const BIITensorInfo *input,
                                 const BIITensorInfo *output,
                                 const BIITensorInfo *axis,
                                 const bool           use_inverted_axis = false);
    };

} // namespace BatmanInfer
