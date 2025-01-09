//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_WRAPPER_CGT_HPP
#define BATMANINFER_WRAPPER_CGT_HPP

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VCGT_IMPL(rtype, vtype, prefix, postfix)      \
    inline rtype vcgt(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

        VCGT_IMPL(uint8x8_t, uint8x8_t, vcgt, u8)

        VCGT_IMPL(uint8x8_t, int8x8_t, vcgt, s8)

        VCGT_IMPL(uint16x4_t, uint16x4_t, vcgt, u16)

        VCGT_IMPL(uint16x4_t, int16x4_t, vcgt, s16)

        VCGT_IMPL(uint32x2_t, uint32x2_t, vcgt, u32)

        VCGT_IMPL(uint32x2_t, int32x2_t, vcgt, s32)

        VCGT_IMPL(uint32x2_t, float32x2_t, vcgt, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCGT_IMPL(uint16x4_t, float16x4_t, vcgt, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCGT_IMPL(uint8x16_t, uint8x16_t, vcgtq, u8)

        VCGT_IMPL(uint8x16_t, int8x16_t, vcgtq, s8)

        VCGT_IMPL(uint16x8_t, uint16x8_t, vcgtq, u16)

        VCGT_IMPL(uint16x8_t, int16x8_t, vcgtq, s16)

        VCGT_IMPL(uint32x4_t, uint32x4_t, vcgtq, u32)

        VCGT_IMPL(uint32x4_t, int32x4_t, vcgtq, s32)

        VCGT_IMPL(uint32x4_t, float32x4_t, vcgtq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCGT_IMPL(uint16x8_t, float16x8_t, vcgtq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCGT_IMPL
    }
}

#endif //BATMANINFER_WRAPPER_CGT_HPP
