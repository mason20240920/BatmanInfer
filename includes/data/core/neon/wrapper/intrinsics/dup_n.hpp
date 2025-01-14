
#ifndef ARM_COMPUTE_WRAPPER_DUP_N_H
#define ARM_COMPUTE_WRAPPER_DUP_N_H

#include <data/core/neon/wrapper/traits.hpp>

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VDUP_N_IMPL(stype, vtype, prefix, postfix, tag) \
    inline vtype vdup_n(stype value, tag)               \
    {                                                   \
        return prefix##_##postfix(value);               \
    }

        VDUP_N_IMPL(uint8_t, uint8x8_t, vdup_n, u8, traits::vector_64_tag)

        VDUP_N_IMPL(int8_t, int8x8_t, vdup_n, s8, traits::vector_64_tag)

        VDUP_N_IMPL(uint16_t, uint16x4_t, vdup_n, u16, traits::vector_64_tag)

        VDUP_N_IMPL(int16_t, int16x4_t, vdup_n, s16, traits::vector_64_tag)

        VDUP_N_IMPL(uint32_t, uint32x2_t, vdup_n, u32, traits::vector_64_tag)

        VDUP_N_IMPL(int32_t, int32x2_t, vdup_n, s32, traits::vector_64_tag)

        VDUP_N_IMPL(float, float32x2_t, vdup_n, f32, traits::vector_64_tag)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VDUP_N_IMPL(float16_t, float16x4_t, vdup_n, f16, traits::vector_64_tag)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VDUP_N_IMPL(uint8_t, uint8x16_t, vdupq_n, u8, traits::vector_128_tag)

        VDUP_N_IMPL(int8_t, int8x16_t, vdupq_n, s8, traits::vector_128_tag)

        VDUP_N_IMPL(uint16_t, uint16x8_t, vdupq_n, u16, traits::vector_128_tag)

        VDUP_N_IMPL(int16_t, int16x8_t, vdupq_n, s16, traits::vector_128_tag)

        VDUP_N_IMPL(uint32_t, uint32x4_t, vdupq_n, u32, traits::vector_128_tag)

        VDUP_N_IMPL(int32_t, int32x4_t, vdupq_n, s32, traits::vector_128_tag)

        VDUP_N_IMPL(float, float32x4_t, vdupq_n, f32, traits::vector_128_tag)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VDUP_N_IMPL(float16_t, float16x8_t, vdupq_n, f16, traits::vector_128_tag)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VDUP_N_IMPL
    } // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_DUP_N_H */
