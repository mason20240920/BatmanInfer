//
// Created by Mason on 2025/4/10.
//

#pragma once

#include <data/core/bi_helpers.hpp>
#include <cpu/kernels/reduction_layer/generic/neon/fp16.hpp>
#include <cpu/kernels/reduction_layer/generic/neon/fp32.hpp>
#include <cpu/kernels/reduction_layer/generic/neon/integer.hpp>
#include <cpu/kernels/reduction_layer/generic/neon/qasymm8_signed.hpp>
#include <cpu/kernels/reduction_layer/generic/neon/qasymm8.hpp>

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_REDUCTION_KERNEL(func_name) \
void func_name(const BIWindow &window, const BIITensor *in, BIITensor *out, const BIReductionOperation op)

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_complex_reduceZ_float32_4_2_SUM);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpX_reduceX_float32_4);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceY_float32_4);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceZ_float32_4);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceW_float32_4);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpX_reduceX_float16_8);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceY_float16_8);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceZ_float16_8);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceW_float16_8);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpX_reduceX_S32_4);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceY_S32_4);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceZ_S32_4);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceW_S32_4);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpX_reduceX_qasymm8);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceY_qasymm8);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceZ_qasymm8);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceW_qasymm8);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpX_reduceX_qasymm8_signed);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceY_qasymm8_signed);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceZ_qasymm8_signed);

        DECLARE_REDUCTION_KERNEL(reduce_RedOpYZW_reduceW_qasymm8_signed);

#undef DECLARE_REDUCTION_KERNEL
    }
}
