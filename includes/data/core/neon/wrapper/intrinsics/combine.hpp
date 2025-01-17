//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_WRAPPER_COMBINE_HPP
#define BATMANINFER_WRAPPER_COMBINE_HPP

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VCOMBINE_IMPL(rtype, vtype, prefix, postfix)      \
    inline rtype vcombine(const vtype &a, const vtype &b) \
    {                                                     \
        return prefix##_##postfix(a, b);                  \
    }

        VCOMBINE_IMPL(uint8x16_t, uint8x8_t, vcombine, u8)

        VCOMBINE_IMPL(int8x16_t, int8x8_t, vcombine, s8)

        VCOMBINE_IMPL(uint16x8_t, uint16x4_t, vcombine, u16)

        VCOMBINE_IMPL(int16x8_t, int16x4_t, vcombine, s16)

        VCOMBINE_IMPL(uint32x4_t, uint32x2_t, vcombine, u32)

        VCOMBINE_IMPL(int32x4_t, int32x2_t, vcombine, s32)

        VCOMBINE_IMPL(float32x4_t, float32x2_t, vcombine, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCOMBINE_IMPL(float16x8_t, float16x4_t, vcombine, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCOMBINE_IMPL
    }
}

#endif //BATMANINFER_WRAPPER_COMBINE_HPP
