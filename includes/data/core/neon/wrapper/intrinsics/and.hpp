//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_WRAPPER_AND_HPP
#define BATMANINFER_WRAPPER_AND_HPP

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VAND_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vand(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

        VAND_IMPL(uint8_t, uint8x8_t, vand, u8)

        VAND_IMPL(int8_t, int8x8_t, vand, s8)

        VAND_IMPL(uint16_t, uint16x4_t, vand, u16)

        VAND_IMPL(int16_t, int16x4_t, vand, s16)

        VAND_IMPL(uint32_t, uint32x2_t, vand, u32)

        VAND_IMPL(int32_t, int32x2_t, vand, s32)

        VAND_IMPL(uint64_t, uint64x1_t, vand, u64)

        VAND_IMPL(int64_t, int64x1_t, vand, s64)

        VAND_IMPL(uint8_t, uint8x16_t, vandq, u8)

        VAND_IMPL(int8_t, int8x16_t, vandq, s8)

        VAND_IMPL(uint16_t, uint16x8_t, vandq, u16)

        VAND_IMPL(int16_t, int16x8_t, vandq, s16)

        VAND_IMPL(uint32_t, uint32x4_t, vandq, u32)

        VAND_IMPL(int32_t, int32x4_t, vandq, s32)

        VAND_IMPL(uint64_t, uint64x2_t, vandq, u64)

        VAND_IMPL(int64_t, int64x2_t, vandq, s64)

#undef VAND_IMPL
    }
}

#endif //BATMANINFER_WRAPPER_AND_HPP
