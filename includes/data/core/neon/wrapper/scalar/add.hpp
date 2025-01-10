//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_WRAPPER_SCALAR_ADD_HPP
#define BATMANINFER_WRAPPER_SCALAR_ADD_HPP

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
        inline uint8_t add_sat(const uint8_t &a, const uint8_t &b) {
            const uint8x8_t va = {a, 0, 0, 0, 0, 0, 0, 0};
            const uint8x8_t vb = {b, 0, 0, 0, 0, 0, 0, 0};
            return vget_lane_u8(vqadd_u8(va, vb), 0);
        }

        inline int16_t add_sat(const int16_t &a, const int16_t &b) {
            const int16x4_t va = {a, 0, 0, 0};
            const int16x4_t vb = {b, 0, 0, 0};
            return vget_lane_s16(vqadd_s16(va, vb), 0);
        }

        inline int32_t add_sat(const int32_t &a, const int32_t &b) {
            const int32x2_t va = {a, 0};
            const int32x2_t vb = {b, 0};
            return vget_lane_s32(vqadd_s32(va, vb), 0);
        }

        inline float add_sat(const float &a, const float &b) {
            // No notion of saturation exists in floating point
            return a + b;
        }

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        inline float16_t add_sat(const float16_t &a, const float16_t &b) {
            // No notion of saturation exists in floating point
            return a + b;
        }

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    }
}

#endif //BATMANINFER_WRAPPER_SCALAR_ADD_HPP
