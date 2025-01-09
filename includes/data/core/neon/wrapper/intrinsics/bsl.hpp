//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_WRAPPER_BSL_HPP
#define BATMANINFER_WRAPPER_BSL_HPP

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VBSL_IMPL(stype, vtype, ctype, prefix, postfix)               \
    inline vtype vbsl(const ctype &a, const vtype &b, const vtype &c) \
    {                                                                 \
        return prefix##_##postfix(a, b, c);                           \
    }

        VBSL_IMPL(uint8_t, uint8x8_t, uint8x8_t, vbsl, u8)

        VBSL_IMPL(int8_t, int8x8_t, uint8x8_t, vbsl, s8)

        VBSL_IMPL(uint16_t, uint16x4_t, uint16x4_t, vbsl, u16)

        VBSL_IMPL(int16_t, int16x4_t, uint16x4_t, vbsl, s16)

        VBSL_IMPL(uint32_t, uint32x2_t, uint32x2_t, vbsl, u32)

        VBSL_IMPL(int32_t, int32x2_t, uint32x2_t, vbsl, s32)

        VBSL_IMPL(float32x2_t, float32x2_t, uint32x2_t, vbsl, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VBSL_IMPL(float16x4_t, float16x4_t, uint16x4_t, vbsl, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VBSL_IMPL(uint8_t, uint8x16_t, uint8x16_t, vbslq, u8)

        VBSL_IMPL(int8_t, int8x16_t, uint8x16_t, vbslq, s8)

        VBSL_IMPL(uint16_t, uint16x8_t, uint16x8_t, vbslq, u16)

        VBSL_IMPL(int16_t, int16x8_t, uint16x8_t, vbslq, s16)

        VBSL_IMPL(uint32_t, uint32x4_t, uint32x4_t, vbslq, u32)

        VBSL_IMPL(int32_t, int32x4_t, uint32x4_t, vbslq, s32)

        VBSL_IMPL(float32x4_t, float32x4_t, uint32x4_t, vbslq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VBSL_IMPL(float16x8_t, float16x8_t, uint16x8_t, vbslq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VBSL_IMPL
    }
}

#endif //BATMANINFER_WRAPPER_BSL_HPP
