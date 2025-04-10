//
// Created by holynova on 25-4-8.
//

#pragma once

namespace BatmanInfer {

namespace cpu {

#define DECLARE_CAST_KERNEL(func_name)                                                                        \
    void func_name(const BIITensor *_src, BIITensor *_dst, const ThreadInfo &tensor, BIConvertPolicy _policy, \
                   const BIWindow &window)

DECLARE_CAST_KERNEL(neon_fp32_to_fp16_cast);
DECLARE_CAST_KERNEL(neon_u8_to_fp16_cast);
DECLARE_CAST_KERNEL(neon_fp16_to_other_dt_cast);
DECLARE_CAST_KERNEL(neon_s32_to_fp16_cast);
DECLARE_CAST_KERNEL(neon_qasymm8_signed_to_fp16_cast);
DECLARE_CAST_KERNEL(neon_fp32_to_bfloat16_cast);
DECLARE_CAST_KERNEL(neon_bfloat16_to_fp32_cast);

#undef DECLARE_CAST_KERNEL

} // namespace cpu

} // namespace BatmanInfer
