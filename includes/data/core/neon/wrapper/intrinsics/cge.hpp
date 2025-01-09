//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_WRAPPER_CGE_HPP
#define BATMANINFER_WRAPPER_CGE_HPP

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VCEQ_IMPL(votype, vtype, prefix, postfix)      \
    inline votype vceq(const vtype &a, const vtype &b) \
    {                                                  \
        return prefix##_##postfix(a, b);               \
    }

        VCEQ_IMPL(uint8x8_t, uint8x8_t, vceq, u8)

        VCEQ_IMPL(uint8x8_t, int8x8_t, vceq, s8)

        VCEQ_IMPL(uint16x4_t, uint16x4_t, vceq, u16)

        VCEQ_IMPL(uint16x4_t, int16x4_t, vceq, s16)

        VCEQ_IMPL(uint32x2_t, uint32x2_t, vceq, u32)

        VCEQ_IMPL(uint32x2_t, int32x2_t, vceq, s32)

        VCEQ_IMPL(uint32x2_t, float32x2_t, vceq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCEQ_IMPL(uint16x4_t, float16x4_t, vceq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCEQ_IMPL(uint8x16_t, uint8x16_t, vceqq, u8)

        VCEQ_IMPL(uint8x16_t, int8x16_t, vceqq, s8)

        VCEQ_IMPL(uint16x8_t, uint16x8_t, vceqq, u16)

        VCEQ_IMPL(uint16x8_t, int16x8_t, vceqq, s16)

        VCEQ_IMPL(uint32x4_t, uint32x4_t, vceqq, u32)

        VCEQ_IMPL(uint32x4_t, int32x4_t, vceqq, s32)

        VCEQ_IMPL(uint32x4_t, float32x4_t, vceqq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCEQ_IMPL(uint16x8_t, float16x8_t, vceqq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCEQ_IMPL
    }
}

#endif //BATMANINFER_WRAPPER_CGE_HPP
