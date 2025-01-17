//
// Created by holynova on 2025/1/16.
//

#pragma once

#include "data/core/bi_types.hpp"
#include "runtime/cpp/bi_i_cpp_simple_function.h"

namespace BatmanInfer {

    class BIITensor;

    /** Basic function to run @ref CPPUpsample */
    class BICPPUpsample : public BIICPPSimpleFunction
    {
    public:
        /** Configure the upsample CPP kernel
         *
         * @param[in]  input  The input tensor to upsample. Data types supported: All.
         * @param[out] output The output tensor. Data types supported: same as @p input
         * @param[in]  info   Padding information
         */
        void configure(const BIITensor *input, BIITensor *output, const BIPadStrideInfo &info);
    };

} // namespace BatmanInfer
