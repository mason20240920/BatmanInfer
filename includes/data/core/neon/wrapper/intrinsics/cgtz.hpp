//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_WRAPPER_CGTZ_HPP
#define BATMANINFER_WRAPPER_CGTZ_HPP

#ifdef __aarch64__

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VCGTZ_IMPL(vtype, rtype, prefix, postfix) \
    inline rtype vcgtz(const vtype &a)            \
    {                                             \
        return prefix##_##postfix(a);             \
    }

        VCGTZ_IMPL(int8x8_t, uint8x8_t, vcgtz, s8)

        VCGTZ_IMPL(int16x4_t, uint16x4_t, vcgtz, s16)

        VCGTZ_IMPL(int32x2_t, uint32x2_t, vcgtz, s32)

        VCGTZ_IMPL(float32x2_t, uint32x2_t, vcgtz, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCGTZ_IMPL(float16x4_t, uint16x4_t, vcgtz, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCGTZ_IMPL(int8x16_t, uint8x16_t, vcgtzq, s8)

        VCGTZ_IMPL(int16x8_t, uint16x8_t, vcgtzq, s16)

        VCGTZ_IMPL(int32x4_t, uint32x4_t, vcgtzq, s32)

        VCGTZ_IMPL(float32x4_t, uint32x4_t, vcgtzq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VCGTZ_IMPL(float16x8_t, uint16x8_t, vcgtzq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VCGTZ_IMPL

    } // namespace wrapper
} // namespace BatmanInfer

#endif // __aarch64__

#endif //BATMANINFER_WRAPPER_CGTZ_HPP
