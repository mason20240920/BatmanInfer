
#include <data/core/neon/bi_ne_asymm.hpp>

namespace BatmanInfer {
    template<BIRoundingPolicy round_policy>
    inline qasymm8x16_t vmlaq_qasymm8(qasymm8x16_t vd, float32x4_t vs, float32x4_t vo) {
        // Convert uint8 vectors to uint16 vectors
        const uint8x8_t vd_low = vget_low_u8(vd);
        const uint8x8_t vd_high = vget_high_u8(vd);
        uint16x8_t vd_low_u16x8 = vmovl_u8(vd_low);
        uint16x8_t vd_high_u16x8 = vmovl_u8(vd_high);
        // Convert uint16 vectors to uint32 vectors
        uint32x4_t A_u32x4 = vmovl_u16(vget_low_u16(vd_low_u16x8));
        uint32x4_t B_u32x4 = vmovl_u16(vget_high_u16(vd_low_u16x8));
        uint32x4_t C_u32x4 = vmovl_u16(vget_low_u16(vd_high_u16x8));
        uint32x4_t D_u32x4 = vmovl_u16(vget_high_u16(vd_high_u16x8));
        // Convert uint32 vectors to float32 vectors
        float32x4_t A_f32x4 = vcvtq_f32_u32(A_u32x4);
        float32x4_t B_f32x4 = vcvtq_f32_u32(B_u32x4);
        float32x4_t C_f32x4 = vcvtq_f32_u32(C_u32x4);
        float32x4_t D_f32x4 = vcvtq_f32_u32(D_u32x4);
        // vd = vd*vs + vo
        A_f32x4 = vmlaq_f32(vo, A_f32x4, vs);
        B_f32x4 = vmlaq_f32(vo, B_f32x4, vs);
        C_f32x4 = vmlaq_f32(vo, C_f32x4, vs);
        D_f32x4 = vmlaq_f32(vo, D_f32x4, vs);
        // Convert float32 vectors to uint32 vectors
#if __aarch64__
        if (round_policy == BIRoundingPolicy::TO_NEAREST_EVEN) {
            A_u32x4 = vcvtnq_u32_f32(A_f32x4);
            B_u32x4 = vcvtnq_u32_f32(B_f32x4);
            C_u32x4 = vcvtnq_u32_f32(C_f32x4);
            D_u32x4 = vcvtnq_u32_f32(D_f32x4);
        } else if (round_policy == BIRoundingPolicy::TO_NEAREST_UP) {
            A_u32x4 = vcvtaq_u32_f32(A_f32x4);
            B_u32x4 = vcvtaq_u32_f32(B_f32x4);
            C_u32x4 = vcvtaq_u32_f32(C_f32x4);
            D_u32x4 = vcvtaq_u32_f32(D_f32x4);
        } else {
            A_u32x4 = vcvtq_u32_f32(A_f32x4);
            B_u32x4 = vcvtq_u32_f32(B_f32x4);
            C_u32x4 = vcvtq_u32_f32(C_f32x4);
            D_u32x4 = vcvtq_u32_f32(D_f32x4);
        }
#else  // #if __aarch64__
        // rounding mode only supported in aarch64
    A_u32x4 = vcvtq_u32_f32(A_f32x4);
    B_u32x4 = vcvtq_u32_f32(B_f32x4);
    C_u32x4 = vcvtq_u32_f32(C_f32x4);
    D_u32x4 = vcvtq_u32_f32(D_f32x4);
#endif // #if __aarch64__
        // Convert uint32 vectors to uint16 vectors (with saturation)
        vd_low_u16x8 = vcombine_u16(vqmovn_u32(A_u32x4), vqmovn_u32(B_u32x4));
        vd_high_u16x8 = vcombine_u16(vqmovn_u32(C_u32x4), vqmovn_u32(D_u32x4));
        // convert uint16 vectors to uint8 vectors (with saturation)
        return vcombine_u8(vqmovn_u16(vd_low_u16x8), vqmovn_u16(vd_high_u16x8));
    }

    template<BIRoundingPolicy round_policy>
    inline qasymm8x16_signed_t vmlaq_qasymm8_signed(qasymm8x16_signed_t vd, float32x4_t vs, float32x4_t vo) {
        // Convert uint8 vectors to int16 vectors
        const int8x8_t vd_low = vget_low_s8(vd);
        const int8x8_t vd_high = vget_high_s8(vd);
        int16x8_t vd_low_s16x8 = vmovl_s8(vd_low);
        int16x8_t vd_high_s16x8 = vmovl_s8(vd_high);
        // Convert int16 vectors to int32 vectors
        int32x4_t A_s32x4 = vmovl_s16(vget_low_s16(vd_low_s16x8));
        int32x4_t B_s32x4 = vmovl_s16(vget_high_s16(vd_low_s16x8));
        int32x4_t C_s32x4 = vmovl_s16(vget_low_s16(vd_high_s16x8));
        int32x4_t D_s32x4 = vmovl_s16(vget_high_s16(vd_high_s16x8));
        // Convert int32 vectors to float32 vectors
        float32x4_t A_f32x4 = vcvtq_f32_s32(A_s32x4);
        float32x4_t B_f32x4 = vcvtq_f32_s32(B_s32x4);
        float32x4_t C_f32x4 = vcvtq_f32_s32(C_s32x4);
        float32x4_t D_f32x4 = vcvtq_f32_s32(D_s32x4);
        // vd = vd*vs + vo
        A_f32x4 = vmlaq_f32(vo, A_f32x4, vs);
        B_f32x4 = vmlaq_f32(vo, B_f32x4, vs);
        C_f32x4 = vmlaq_f32(vo, C_f32x4, vs);
        D_f32x4 = vmlaq_f32(vo, D_f32x4, vs);
#if __aarch64__
        if (round_policy == BIRoundingPolicy::TO_NEAREST_EVEN) {
            A_s32x4 = vcvtnq_s32_f32(A_f32x4);
            B_s32x4 = vcvtnq_s32_f32(B_f32x4);
            C_s32x4 = vcvtnq_s32_f32(C_f32x4);
            D_s32x4 = vcvtnq_s32_f32(D_f32x4);
        } else if (round_policy == BIRoundingPolicy::TO_NEAREST_UP) {
            A_s32x4 = vcvtaq_s32_f32(A_f32x4);
            B_s32x4 = vcvtaq_s32_f32(B_f32x4);
            C_s32x4 = vcvtaq_s32_f32(C_f32x4);
            D_s32x4 = vcvtaq_s32_f32(D_f32x4);
        } else {
            A_s32x4 = vcvtq_s32_f32(A_f32x4);
            B_s32x4 = vcvtq_s32_f32(B_f32x4);
            C_s32x4 = vcvtq_s32_f32(C_f32x4);
            D_s32x4 = vcvtq_s32_f32(D_f32x4);
        }
#else  // #if __aarch64__
        // rounding mode only supported in aarch64
    A_s32x4 = vcvtq_s32_f32(A_f32x4);
    B_s32x4 = vcvtq_s32_f32(B_f32x4);
    C_s32x4 = vcvtq_s32_f32(C_f32x4);
    D_s32x4 = vcvtq_s32_f32(D_f32x4);
#endif // #if __aarch64__

        // Convert int32 vectors to int16 vectors (with saturation)
        vd_low_s16x8 = vcombine_s16(vqmovn_s32(A_s32x4), vqmovn_s32(B_s32x4));
        vd_high_s16x8 = vcombine_s16(vqmovn_s32(C_s32x4), vqmovn_s32(D_s32x4));
        // convert int16 vectors to int8 vectors (with saturation)
        return vcombine_s8(vqmovn_s16(vd_low_s16x8), vqmovn_s16(vd_high_s16x8));
    }
}