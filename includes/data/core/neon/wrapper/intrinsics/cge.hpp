//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_WRAPPER_CGE_HPP
#define BATMANINFER_WRAPPER_CGE_HPP

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VCGE_IMPL(stype, vtype, rtype, prefix, postfix) \
    inline rtype vcge(const vtype &a, const vtype &b)   \
    {                                                   \
        return prefix##_##postfix(a, b);                \
    }

        VCGE_IMPL(uint8_t, uint8x8_t, uint8x8_t, vcge, u8)

        VCGE_IMPL(int8_t, int8x8_t, uint8x8_t, vcge, s8)

        VCGE_IMPL(uint16_t, uint16x4_t, uint16x4_t, vcge, u16)

        VCGE_IMPL(int16_t, int16x4_t, uint16x4_t, vcge, s16)

        VCGE_IMPL(uint32_t, uint32x2_t, uint32x2_t, vcge, u32)

        VCGE_IMPL(int32_t, int32x2_t, uint32x2_t, vcge, s32)

        VCGE_IMPL(float32x2_t, float32x2_t, uint32x2_t, vcge, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCGE_IMPL(float16x4_t, float16x4_t, uint16x4_t, vcge, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCGE_IMPL(uint8_t, uint8x16_t, uint8x16_t, vcgeq, u8)

        VCGE_IMPL(int8_t, int8x16_t, uint8x16_t, vcgeq, s8)

        VCGE_IMPL(uint16_t, uint16x8_t, uint16x8_t, vcgeq, u16)

        VCGE_IMPL(int16_t, int16x8_t, uint16x8_t, vcgeq, s16)

        VCGE_IMPL(uint32_t, uint32x4_t, uint32x4_t, vcgeq, u32)

        VCGE_IMPL(int32_t, int32x4_t, uint32x4_t, vcgeq, s32)

        VCGE_IMPL(float32x4_t, float32x4_t, uint32x4_t, vcgeq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCGE_IMPL(float16x8_t, float16x8_t, uint16x8_t, vcgeq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCGE_IMPL
    }
}

#endif //BATMANINFER_WRAPPER_CGE_HPP
