//
// Created by Mason on 2025/4/3.
//

#pragma once

#include <cpu/kernels/select/generic/neon/fp16.hpp>
#include <cpu/kernels/select/generic/neon/fp32.hpp>
#include <cpu/kernels/select/generic/neon/integer.hpp>

namespace BatmanInfer::cpu {
#define DECLARE_SELECT_KERNEL(func_name) \
void func_name(const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output, const BIWindow &window)

    DECLARE_SELECT_KERNEL(neon_s8_select_same_rank);

    DECLARE_SELECT_KERNEL(neon_s16_select_same_rank);

    DECLARE_SELECT_KERNEL(neon_s32_select_same_rank);

    DECLARE_SELECT_KERNEL(neon_u8_select_same_rank);

    DECLARE_SELECT_KERNEL(neon_u16_select_same_rank);

    DECLARE_SELECT_KERNEL(neon_u32_select_same_rank);

    DECLARE_SELECT_KERNEL(neon_f16_select_same_rank);

    DECLARE_SELECT_KERNEL(neon_f32_select_same_rank);

    DECLARE_SELECT_KERNEL(neon_s8_select_not_same_rank);

    DECLARE_SELECT_KERNEL(neon_s16_select_not_same_rank);

    DECLARE_SELECT_KERNEL(neon_s32_select_not_same_rank);

    DECLARE_SELECT_KERNEL(neon_u8_select_not_same_rank);

    DECLARE_SELECT_KERNEL(neon_u16_select_not_same_rank);

    DECLARE_SELECT_KERNEL(neon_u32_select_not_same_rank);

    DECLARE_SELECT_KERNEL(neon_f16_select_not_same_rank);

    DECLARE_SELECT_KERNEL(neon_f32_select_not_same_rank);

#undef DECLARE_RANGE_KERNEL
} // namespace cpu
