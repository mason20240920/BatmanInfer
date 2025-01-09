//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_WRAPPER_CLT_HPP
#define BATMANINFER_WRAPPER_CLT_HPP

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VCLT_IMPL(votype, vtype, prefix, postfix)      \
    inline votype vclt(const vtype &a, const vtype &b) \
    {                                                  \
        return prefix##_##postfix(a, b);               \
    }

        VCLT_IMPL(uint8x8_t, uint8x8_t, vclt, u8)

        VCLT_IMPL(uint8x8_t, int8x8_t, vclt, s8)

        VCLT_IMPL(uint16x4_t, uint16x4_t, vclt, u16)

        VCLT_IMPL(uint16x4_t, int16x4_t, vclt, s16)

        VCLT_IMPL(uint32x2_t, uint32x2_t, vclt, u32)

        VCLT_IMPL(uint32x2_t, int32x2_t, vclt, s32)

        VCLT_IMPL(uint32x2_t, float32x2_t, vclt, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCLT_IMPL(uint16x4_t, float16x4_t, vclt, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCLT_IMPL(uint8x16_t, uint8x16_t, vcltq, u8)

        VCLT_IMPL(uint8x16_t, int8x16_t, vcltq, s8)

        VCLT_IMPL(uint16x8_t, uint16x8_t, vcltq, u16)

        VCLT_IMPL(uint16x8_t, int16x8_t, vcltq, s16)

        VCLT_IMPL(uint32x4_t, uint32x4_t, vcltq, u32)

        VCLT_IMPL(uint32x4_t, int32x4_t, vcltq, s32)

        VCLT_IMPL(uint32x4_t, float32x4_t, vcltq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCLT_IMPL(uint16x8_t, float16x8_t, vcltq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCLT_IMPL
    }
}

#endif //BATMANINFER_WRAPPER_CLT_HPP
