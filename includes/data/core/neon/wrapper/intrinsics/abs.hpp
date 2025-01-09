//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_WRAPPER_ABS_HPP
#define BATMANINFER_WRAPPER_ABS_HPP

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VABS_IMPL(stype, vtype, prefix, postfix) \
    inline vtype vabs(const vtype &a)            \
    {                                            \
        return prefix##_##postfix(a);            \
    }

#define VQABS_IMPL(stype, vtype, prefix, postfix) \
    inline vtype vqabs(const vtype &a)            \
    {                                             \
        return prefix##_##postfix(a);             \
    }

// Absolute: vabs{q}_<type>. Vd[i] = |Va[i]|
        VABS_IMPL(int8x8_t, int8x8_t, vabs, s8)

        VABS_IMPL(int16x4_t, int16x4_t, vabs, s16)

        VABS_IMPL(int32x2_t, int32x2_t, vabs, s32)

        VABS_IMPL(float32x2_t, float32x2_t, vabs, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VABS_IMPL(float16x4_t, float16x4_t, vabs, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VABS_IMPL(int8x16_t, int8x16_t, vabsq, s8)

        VABS_IMPL(int16x8_t, int16x8_t, vabsq, s16)

        VABS_IMPL(int32x4_t, int32x4_t, vabsq, s32)

        VABS_IMPL(float32x4_t, float32x4_t, vabsq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VABS_IMPL(float16x8_t, float16x8_t, vabsq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

// Saturating absolute: vqabs{q}_<type>. Vd[i] = sat(|Va[i]|)
        VQABS_IMPL(int8x8_t, int8x8_t, vqabs, s8)

        VQABS_IMPL(int16x4_t, int16x4_t, vqabs, s16)

        VQABS_IMPL(int32x2_t, int32x2_t, vqabs, s32)

        VQABS_IMPL(int8x16_t, int8x16_t, vqabsq, s8)

        VQABS_IMPL(int16x8_t, int16x8_t, vqabsq, s16)

        VQABS_IMPL(int32x4_t, int32x4_t, vqabsq, s32)

#undef VABS_IMPL
#undef VQABS_IMPL
    }
}

#endif //BATMANINFER_WRAPPER_ABS_HPP
