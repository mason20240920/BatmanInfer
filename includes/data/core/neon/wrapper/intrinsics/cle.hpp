//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_WRAPPER_CLE_HPP
#define BATMANINFER_WRAPPER_CLE_HPP

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VCLE_IMPL(stype, vtype, rtype, prefix, postfix) \
    inline rtype vcle(const vtype &a, const vtype &b)   \
    {                                                   \
        return prefix##_##postfix(a, b);                \
    }

        VCLE_IMPL(uint8_t, uint8x8_t, uint8x8_t, vcle, u8)

        VCLE_IMPL(int8_t, int8x8_t, uint8x8_t, vcle, s8)

        VCLE_IMPL(uint16_t, uint16x4_t, uint16x4_t, vcle, u16)

        VCLE_IMPL(int16_t, int16x4_t, uint16x4_t, vcle, s16)

        VCLE_IMPL(uint32_t, uint32x2_t, uint32x2_t, vcle, u32)

        VCLE_IMPL(int32_t, int32x2_t, uint32x2_t, vcle, s32)

        VCLE_IMPL(float32x2_t, float32x2_t, uint32x2_t, vcle, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCLE_IMPL(float16x4_t, float16x4_t, uint16x4_t, vcle, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCLE_IMPL(uint8_t, uint8x16_t, uint8x16_t, vcleq, u8)

        VCLE_IMPL(int8_t, int8x16_t, uint8x16_t, vcleq, s8)

        VCLE_IMPL(uint16_t, uint16x8_t, uint16x8_t, vcleq, u16)

        VCLE_IMPL(int16_t, int16x8_t, uint16x8_t, vcleq, s16)

        VCLE_IMPL(uint32_t, uint32x4_t, uint32x4_t, vcleq, u32)

        VCLE_IMPL(int32_t, int32x4_t, uint32x4_t, vcleq, s32)

        VCLE_IMPL(float32x4_t, float32x4_t, uint32x4_t, vcleq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCLE_IMPL(float16x8_t, float16x8_t, uint16x8_t, vcleq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCLE_IMPL
    } // namespace wrapper
}

#endif //BATMANINFER_WRAPPER_CLE_HPP
