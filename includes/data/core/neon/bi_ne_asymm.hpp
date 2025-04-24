//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BI_NE_ASYMM_HPP
#define BATMANINFER_BI_NE_ASYMM_HPP

#include <data/core/quantization_info.hpp>
#include <data/core/bi_rounding.h>

#include <data/core/neon/bi_neon_math.hpp>
#include <data/core/neon/wrapper/intrinsics/intrinsics.h>

#include <neon/neon_defines.h>

namespace BatmanInfer {
    using qasymm8x8_t = uint8x8_t; /**< 8位量化非对称向量，包含8个元素 */
    using qasymm8x8x2_t = uint8x8x2_t; /**< 8位量化非对称向量，包含16个元素 */
    using qasymm8x8x3_t = uint8x8x3_t; /**< 8位量化非对称向量，包含24个元素 */
    using qasymm8x8x4_t = uint8x8x4_t; /**< 8位量化非对称向量，包含32个元素 */
    using qasymm8x16_t = uint8x16_t; /**< 8位量化非对称向量，包含16个元素 */

    using qasymm8x8_signed_t = int8x8_t; /**< 8位量化有符号非对称向量，包含8个元素 */
    using qasymm8x8x2_signed_t = int8x8x2_t; /**< 8位量化有符号非对称向量，包含16个元素 */
    using qasymm8x8x3_signed_t = int8x8x3_t; /**< 8位量化有符号非对称向量，包含24个元素 */
    using qasymm8x8x4_signed_t = int8x8x4_t; /**< 8位量化有符号非对称向量，包含32个元素 */
    using qasymm8x16_signed_t = int8x16_t; /**< 8位量化有符号非对称向量，包含16个元素 */

    /**
     *
     * 对QASYMM8向量的所有16个分量执行乘法累加操作
     * vd*vs + vo
     *
     * @tparam round_policy
     * @param vd 输入向量值，格式为QASYMM8
     * @param vs 向量乘数，格式为F32。乘数值必须在所有四个通道中重复。
     * @param vo 向量加数，格式为F32。加数值必须在所有四个通道中重复。
     *
     * @return 返回一个QASYMM8格式的16分量向量，结果会被饱和到适合范围内
     */
    template<BIRoundingPolicy round_policy = BIRoundingPolicy::TO_ZERO>
    qasymm8x16_t vmlaq_qasymm8(qasymm8x16_t vd, float32x4_t vs, float32x4_t vo);


    /** Perform a multiply-accumulate on all 16 components of a QASYMM8_SIGNED vector
 *
 * vd*vs + vo
 *
 * @param[in] vd Input vector value in QASYMM8_SIGNED format
 * @param[in] vs Vector multiplier in F32 format. The multiplier value must be duplicated across all four lanes.
 * @param[in] vo Vector addend in F32 format. The addend value must be duplicated across all four lanes.
 *
 * @return A 16-component vector in QASYMM8_SIGNED format, saturated to fit
 */
    template<BIRoundingPolicy round_policy = BIRoundingPolicy::TO_ZERO>
    qasymm8x16_signed_t vmlaq_qasymm8_signed(qasymm8x16_signed_t vd, float32x4_t vs, float32x4_t vo);

    /** Performs final quantization step on 16 elements
     *
     * @param[in] in_s32                        Input to be quantized.
     * @param[in] result_fixedpoint_multiplier  Result multiplier parameter
     * @param[in] result_shift                  Result shift parameter
     * @param[in] result_offset_after_shift_s32 Result offset parameter
     * @param[in] min_u8                        Relu lower bound
     * @param[in] max_u8                        Relu upper bound
     * @param[in] is_bounded_relu               Specified if a fused bounded relu should be applied
     *
     * @return Quantized values
     */
    inline uint8x16_t finalize_quantization(int32x4x4_t &in_s32,
                                            int result_fixedpoint_multiplier,
                                            int32_t result_shift,
                                            int32x4_t result_offset_after_shift_s32,
                                            uint8x16_t min_u8,
                                            uint8x16_t max_u8,
                                            bool is_bounded_relu) {
        const static int32x4_t zero_s32 = vdupq_n_s32(0);

        if (result_shift < 0) {
            in_s32.val[0] = vmulq_n_s32(in_s32.val[0], (1 << (-result_shift)));
            in_s32.val[1] = vmulq_n_s32(in_s32.val[1], (1 << (-result_shift)));
            in_s32.val[2] = vmulq_n_s32(in_s32.val[2], (1 << (-result_shift)));
            in_s32.val[3] = vmulq_n_s32(in_s32.val[3], (1 << (-result_shift)));

            in_s32.val[0] = vqdmulhq_n_s32(in_s32.val[0], result_fixedpoint_multiplier);
            in_s32.val[1] = vqdmulhq_n_s32(in_s32.val[1], result_fixedpoint_multiplier);
            in_s32.val[2] = vqdmulhq_n_s32(in_s32.val[2], result_fixedpoint_multiplier);
            in_s32.val[3] = vqdmulhq_n_s32(in_s32.val[3], result_fixedpoint_multiplier);
        } else {
            // Fixed point multiplication with vector saturating rounding doubling multiply high with scalar
            in_s32.val[0] = vqdmulhq_n_s32(in_s32.val[0], result_fixedpoint_multiplier);
            in_s32.val[1] = vqdmulhq_n_s32(in_s32.val[1], result_fixedpoint_multiplier);
            in_s32.val[2] = vqdmulhq_n_s32(in_s32.val[2], result_fixedpoint_multiplier);
            in_s32.val[3] = vqdmulhq_n_s32(in_s32.val[3], result_fixedpoint_multiplier);

            // Round to the nearest division by a power-of-two using result_shift_s32
            in_s32.val[0] = rounding_divide_by_pow2(in_s32.val[0], result_shift);
            in_s32.val[1] = rounding_divide_by_pow2(in_s32.val[1], result_shift);
            in_s32.val[2] = rounding_divide_by_pow2(in_s32.val[2], result_shift);
            in_s32.val[3] = rounding_divide_by_pow2(in_s32.val[3], result_shift);
        }

        // Add the offset terms
        in_s32.val[0] = vaddq_s32(in_s32.val[0], result_offset_after_shift_s32);
        in_s32.val[1] = vaddq_s32(in_s32.val[1], result_offset_after_shift_s32);
        in_s32.val[2] = vaddq_s32(in_s32.val[2], result_offset_after_shift_s32);
        in_s32.val[3] = vaddq_s32(in_s32.val[3], result_offset_after_shift_s32);

        // Saturate negative values
        in_s32.val[0] = vmaxq_s32(in_s32.val[0], zero_s32);
        in_s32.val[1] = vmaxq_s32(in_s32.val[1], zero_s32);
        in_s32.val[2] = vmaxq_s32(in_s32.val[2], zero_s32);
        in_s32.val[3] = vmaxq_s32(in_s32.val[3], zero_s32);

        // Convert S32 to S16
        const int16x8x2_t in_s16 = {
            {
                vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
            }
        };

        // Convert S16 to U8
        uint8x16_t out_u8 = vcombine_u8(vqmovun_s16(in_s16.val[0]), vqmovun_s16(in_s16.val[1]));

        if (is_bounded_relu) {
            out_u8 = vmaxq_u8(out_u8, min_u8);
            out_u8 = vminq_u8(out_u8, max_u8);
        }

        return out_u8;
    }

    /** Performs final quantization step on 16 elements
     *
     * 对int32x4x4_t的输入做定点量化，输出int8x16_t
     *
     * @param[in] in_s32                        Input to be quantized. 16个int32，4组每组4个
     * @param[in] result_fixedpoint_multiplier  Result multiplier parameter 量化乘法器（单通道/全局）
     * @param[in] result_shift                  Result shift parameter 量化右移位数（单通道/全局）
     * @param[in] result_offset_after_shift_s32 Result offset parameter 量化偏置（单通道/全局）
     * @param[in] min_s8                        Relu lower bound 输出最小值（用于ReLU等裁剪）
     * @param[in] max_s8                        Relu upper bound 输出最大值（用于ReLU等裁剪）
     * @param[in] is_bounded_relu               Specified if a fused bounded relu should be applied 是否启用ReLU裁剪
     *
     * @return Quantized values
     */
    inline int8x16_t finalize_quantization(int32x4x4_t &in_s32,
                                           int result_fixedpoint_multiplier,
                                           int32_t result_shift,
                                           int32x4_t result_offset_after_shift_s32,
                                           int8x16_t min_s8,
                                           int8x16_t max_s8,
                                           bool is_bounded_relu) {
        // 判断是左移还是右移
        if (result_shift < 0) {
            // 先左移
            in_s32.val[0] = vmulq_n_s32(in_s32.val[0], (1 << (-result_shift)));
            in_s32.val[1] = vmulq_n_s32(in_s32.val[1], (1 << (-result_shift)));
            in_s32.val[2] = vmulq_n_s32(in_s32.val[2], (1 << (-result_shift)));
            in_s32.val[3] = vmulq_n_s32(in_s32.val[3], (1 << (-result_shift)));

            // 再做定点乘法（saturating rounding doubling multiply high，NEON指令）
            in_s32.val[0] = vqdmulhq_n_s32(in_s32.val[0], result_fixedpoint_multiplier);
            in_s32.val[1] = vqdmulhq_n_s32(in_s32.val[1], result_fixedpoint_multiplier);
            in_s32.val[2] = vqdmulhq_n_s32(in_s32.val[2], result_fixedpoint_multiplier);
            in_s32.val[3] = vqdmulhq_n_s32(in_s32.val[3], result_fixedpoint_multiplier);
        } else {
            // 先做定点乘法, // 4 个 32 位有符号整数的向量, 一个 32 位有符号整数（标量，广播到每个元素）
            in_s32.val[0] = vqdmulhq_n_s32(in_s32.val[0], result_fixedpoint_multiplier);
            in_s32.val[1] = vqdmulhq_n_s32(in_s32.val[1], result_fixedpoint_multiplier);
            in_s32.val[2] = vqdmulhq_n_s32(in_s32.val[2], result_fixedpoint_multiplier);
            in_s32.val[3] = vqdmulhq_n_s32(in_s32.val[3], result_fixedpoint_multiplier);

            // 再右移（四舍五入除以2^shift）
            in_s32.val[0] = rounding_divide_by_pow2(in_s32.val[0], result_shift);
            in_s32.val[1] = rounding_divide_by_pow2(in_s32.val[1], result_shift);
            in_s32.val[2] = rounding_divide_by_pow2(in_s32.val[2], result_shift);
            in_s32.val[3] = rounding_divide_by_pow2(in_s32.val[3], result_shift);
        }

        // 加上偏置（通常是zero_point+output_offset）
        in_s32.val[0] = vaddq_s32(in_s32.val[0], result_offset_after_shift_s32);
        in_s32.val[1] = vaddq_s32(in_s32.val[1], result_offset_after_shift_s32);
        in_s32.val[2] = vaddq_s32(in_s32.val[2], result_offset_after_shift_s32);
        in_s32.val[3] = vaddq_s32(in_s32.val[3], result_offset_after_shift_s32);

        // int32转int16，先每组4个int32降为4个int16，再拼成8个int16
        const int16x8x2_t in_s16 = {
            {
                vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
            }
        };

        // int16转int8，最终得到16个int8
        int8x16_t out_s8 = vcombine_s8(vqmovn_s16(in_s16.val[0]), vqmovn_s16(in_s16.val[1]));

        // 可选ReLU裁剪
        if (is_bounded_relu) {
            out_s8 = vmaxq_s8(out_s8, min_s8); // 大于最小值
            out_s8 = vminq_s8(out_s8, max_s8); // 小于最大值
        }

        // 返回最终量化结果
        return out_s8;
    }

    /** Performs final quantization step on 16 elements for symmetric quantization
     *
     * @param[in] in_s32                        Input to be quantized.
     * @param[in] result_fixedpoint_multiplier  Result multiplier parameter
     * @param[in] result_shift                  Result shift parameter
     * @param[in] result_offset_after_shift_s32 Result offset parameter
     * @param[in] min_s8                        Relu lower bound
     * @param[in] max_s8                        Relu upper bound
     * @param[in] is_bounded_relu               Specified if a fused bounded relu should be applied
     *
     * @return Quantized values
     */
    inline int8x16_t finalize_quantization_symm(int32x4x4_t &in_s32,
                                                const int32x4x4_t &result_fixedpoint_multiplier,
                                                const int32x4x4_t &result_shift,
                                                const int32x4_t &result_offset_after_shift_s32,
                                                const int8x16_t &min_s8,
                                                const int8x16_t &max_s8,
                                                const bool is_bounded_relu) {
        const static int32x4_t one_s32 = vdupq_n_s32(1);

        // Fixed point multiplication with vector saturating rounding doubling multiply high with scalar
        int32x4x4_t res_shift_gt0 = {
            vqdmulhq_s32(in_s32.val[0], result_fixedpoint_multiplier.val[0]),
            vqdmulhq_s32(in_s32.val[1], result_fixedpoint_multiplier.val[1]),
            vqdmulhq_s32(in_s32.val[2], result_fixedpoint_multiplier.val[2]),
            vqdmulhq_s32(in_s32.val[3], result_fixedpoint_multiplier.val[3]),
        };
        // Round to the nearest division by a power-of-two using result_shift_s32
        res_shift_gt0.val[0] = rounding_divide_by_pow2(res_shift_gt0.val[0], result_shift.val[0]);
        res_shift_gt0.val[1] = rounding_divide_by_pow2(res_shift_gt0.val[1], result_shift.val[1]);
        res_shift_gt0.val[2] = rounding_divide_by_pow2(res_shift_gt0.val[2], result_shift.val[2]);
        res_shift_gt0.val[3] = rounding_divide_by_pow2(res_shift_gt0.val[3], result_shift.val[3]);

        int32x4x4_t res_shift_lt0 = {
            vmulq_s32(in_s32.val[0], vshlq_s32(one_s32, vnegq_s32(result_shift.val[0]))),
            vmulq_s32(in_s32.val[1], vshlq_s32(one_s32, vnegq_s32(result_shift.val[1]))),
            vmulq_s32(in_s32.val[2], vshlq_s32(one_s32, vnegq_s32(result_shift.val[2]))),
            vmulq_s32(in_s32.val[3], vshlq_s32(one_s32, vnegq_s32(result_shift.val[3]))),
        };
        res_shift_lt0.val[0] = vqdmulhq_s32(res_shift_lt0.val[0], result_fixedpoint_multiplier.val[0]);
        res_shift_lt0.val[1] = vqdmulhq_s32(res_shift_lt0.val[1], result_fixedpoint_multiplier.val[1]);
        res_shift_lt0.val[2] = vqdmulhq_s32(res_shift_lt0.val[2], result_fixedpoint_multiplier.val[2]);
        res_shift_lt0.val[3] = vqdmulhq_s32(res_shift_lt0.val[3], result_fixedpoint_multiplier.val[3]);

        // Select result depending on shift value
        const uint32x4x4_t mask_lt0 = {
#ifdef __aarch64__
            vcltzq_s32(result_shift.val[0]),
            vcltzq_s32(result_shift.val[1]),
            vcltzq_s32(result_shift.val[2]),
            vcltzq_s32(result_shift.val[3]),
#else  //__aarch64__
                vcltq_s32(result_shift.val[0], vdupq_n_s32(0)),
        vcltq_s32(result_shift.val[1], vdupq_n_s32(0)),
        vcltq_s32(result_shift.val[2], vdupq_n_s32(0)),
        vcltq_s32(result_shift.val[3], vdupq_n_s32(0)),
#endif //__aarch64__
        };

        in_s32.val[0] = vbslq_s32(mask_lt0.val[0], res_shift_lt0.val[0], res_shift_gt0.val[0]);
        in_s32.val[1] = vbslq_s32(mask_lt0.val[1], res_shift_lt0.val[1], res_shift_gt0.val[1]);
        in_s32.val[2] = vbslq_s32(mask_lt0.val[2], res_shift_lt0.val[2], res_shift_gt0.val[2]);
        in_s32.val[3] = vbslq_s32(mask_lt0.val[3], res_shift_lt0.val[3], res_shift_gt0.val[3]);

        // Add the offset terms
        in_s32.val[0] = vaddq_s32(in_s32.val[0], result_offset_after_shift_s32);
        in_s32.val[1] = vaddq_s32(in_s32.val[1], result_offset_after_shift_s32);
        in_s32.val[2] = vaddq_s32(in_s32.val[2], result_offset_after_shift_s32);
        in_s32.val[3] = vaddq_s32(in_s32.val[3], result_offset_after_shift_s32);

        // Convert S32 to S16
        const int16x8x2_t in_s16 = {
            {
                vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
            }
        };

        // Convert S16 to S8
        int8x16_t out_s8 = vcombine_s8(vqmovn_s16(in_s16.val[0]), vqmovn_s16(in_s16.val[1]));

        if (is_bounded_relu) {
            out_s8 = vmaxq_s8(out_s8, min_s8);
            out_s8 = vminq_s8(out_s8, max_s8);
        }

        return out_s8;
    }

    /** Performs final quantization step on single element
     *
     * @param[in] in_value                      Input to be quantized.
     * @param[in] result_fixedpoint_multiplier  Result multiplier parameter
     * @param[in] result_shift                  Result shift parameter
     * @param[in] result_offset_after_shift_s32 Result offset parameter
     * @param[in] min_u8                        Relu lower bound
     * @param[in] max_u8                        Relu upper bound
     * @param[in] is_bounded_relu               Specified if a fused bounded relu should be applied
     *
     * @return Quantized value
     */
    inline uint8_t finalize_quantization(int32_t in_value,
                                         int result_fixedpoint_multiplier,
                                         int32_t result_shift,
                                         int32_t result_offset_after_shift_s32,
                                         uint8_t min_u8,
                                         uint8_t max_u8,
                                         bool is_bounded_relu) {
        int32x4_t in_s32 = vdupq_n_s32(in_value);

        if (result_shift < 0) {
            in_value = vgetq_lane_s32(
                vqdmulhq_n_s32(vmulq_n_s32(in_s32, (1 << (-result_shift))), result_fixedpoint_multiplier), 0);
        } else {
            // Fixed point multiplication with vector saturating rounding doubling multiply high with scalar
            in_value = vgetq_lane_s32(vqdmulhq_n_s32(in_s32, result_fixedpoint_multiplier), 0);
            // Shift value by result_shift_s32
            in_value = rounding_divide_by_pow2(in_value, result_shift);
        }

        // Add the offset term
        in_value += result_offset_after_shift_s32;

        // Bound the result
        uint8_t out_u8 = static_cast<uint8_t>(std::max<int32_t>(0, std::min<int32_t>(255, in_value)));
        if (is_bounded_relu) {
            out_u8 = static_cast<uint8_t>(std::max(min_u8, std::min(max_u8, out_u8)));
        }

        return out_u8;
    }

    /** Performs final quantization step on single element
     *
     * @param[in] in_value                      Input to be quantized.
     * @param[in] result_fixedpoint_multiplier  Result multiplier parameter
     * @param[in] result_shift                  Result shift parameter
     * @param[in] result_offset_after_shift_s32 Result offset parameter
     * @param[in] min_s8                        Relu lower bound
     * @param[in] max_s8                        Relu upper bound
     * @param[in] is_bounded_relu               Specified if a fused bounded relu should be applied
     *
     * @return Quantized value
     */
    inline int8_t finalize_quantization(int32_t in_value,
                                        int result_fixedpoint_multiplier,
                                        int32_t result_shift,
                                        int32_t result_offset_after_shift_s32,
                                        int8_t min_s8,
                                        int8_t max_s8,
                                        bool is_bounded_relu) {
        int32x4_t in_s32 = vdupq_n_s32(in_value);

        if (result_shift < 0) {
            in_value = vgetq_lane_s32(
                vqdmulhq_n_s32(vmulq_n_s32(in_s32, (1 << (-result_shift))), result_fixedpoint_multiplier), 0);
        } else {
            // Fixed point multiplication with vector saturating rounding doubling multiply high with scalar
            in_value = vgetq_lane_s32(vqdmulhq_n_s32(in_s32, result_fixedpoint_multiplier), 0);

            // Shift value by result_shift_s32
            in_value = rounding_divide_by_pow2(in_value, result_shift);
        }

        // Add the offset term
        in_value += result_offset_after_shift_s32;

        // Bound the result
        int8_t out_s8 = static_cast<int8_t>(std::max<int32_t>(-128, std::min<int32_t>(127, in_value)));
        if (is_bounded_relu) {
            out_s8 = static_cast<int8_t>(std::max(min_s8, std::min(max_s8, out_s8)));
        }

        return out_s8;
    }

    /** Dequantize a neon vector holding 8 quantized values.
     *
     * @param[in] qv Input values to be dequantized.
     * @param[in] qi Quantization information to be used in the computation.
     *
     * @return Dequantized values in a neon vector
     */
    inline float32x4x2_t vdequantize(const uint8x8_t &qv, const BIUniformQuantizationInfo &qi) {
        const float scale = qi.scale;
        const int offset = qi.offset;
        const int32x4_t voffset = vdupq_n_s32(offset);
        const float32x4_t vscale = vdupq_n_f32(scale);
        const float32x4x2_t vdequantized_input = {
            {
                vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(
                                                      vmovl_u16(vget_low_u16(vmovl_u8(qv)))), voffset)),
                          vscale),
                vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(
                                                      vmovl_u16(vget_high_u16(vmovl_u8(qv)))), voffset)),
                          vscale),
            }
        };
        return vdequantized_input;
    }

    /** Dequantize a neon vector holding 8 singed quantized values.
     *
     * @param[in] qv Input values to be dequantized.
     * @param[in] qi Quantization information to be used in the computation.
     *
     * @return Dequantized values in a neon vector
     */
    inline float32x4x2_t vdequantize(const int8x8_t &qv, const BIUniformQuantizationInfo &qi) {
        const float scale = qi.scale;
        const int offset = qi.offset;
        const int32x4_t voffset = vdupq_n_s32(offset);
        const float32x4_t vscale = vdupq_n_f32(scale);
        const float32x4x2_t vdequantized_input = {
            {
                vmulq_f32(vcvtq_f32_s32(
                              vsubq_s32(vmovl_s16(vget_low_s16(vmovl_s8(qv))),
                                        voffset)), vscale),
                vmulq_f32(vcvtq_f32_s32(
                              vsubq_s32(vmovl_s16(vget_high_s16(vmovl_s8(qv))),
                                        voffset)), vscale),
            }
        };
        return vdequantized_input;
    }

    /** Dequantize a neon vector holding 16 quantized values.
     *
     * @param[in] qv      Input values to be dequantized.
     * @param[in] voffset Vectorized dequantization offset.
     * @param[in] vscale  Vectorized dequantization scale.
     *
     * @return Dequantized values in a neon vector
     */
    inline float32x4x4_t vdequantize(const uint8x16_t &qv, const int32x4_t &voffset, const float32x4_t &vscale) {
        const float32x4x4_t vdequantized_input = {
            {
                vmulq_f32(vcvtq_f32_s32(
                              vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                                            vget_low_u16(vmovl_u8(vget_low_u8(qv))))),
                                        voffset)),
                          vscale),
                vmulq_f32(vcvtq_f32_s32(
                              vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                                            vget_high_u16(vmovl_u8(vget_low_u8(qv))))),
                                        voffset)),
                          vscale),
                vmulq_f32(vcvtq_f32_s32(
                              vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                                            vget_low_u16(vmovl_u8(vget_high_u8(qv))))),
                                        voffset)),
                          vscale),
                vmulq_f32(vcvtq_f32_s32(
                              vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                                            vget_high_u16(vmovl_u8(vget_high_u8(qv))))),
                                        voffset)),
                          vscale),
            }
        };
        return vdequantized_input;
    }

    /** Dequantize a neon vector holding 16 quantized values.
     *
     * @param[in] qv Input values to be dequantized.
     * @param[in] qi Quantization information to be used in the computation.
     *
     * @return Dequantized values in a neon vector
     */
    inline float32x4x4_t vdequantize(const uint8x16_t &qv, const BIUniformQuantizationInfo &qi) {
        const float scale = qi.scale;
        const int offset = qi.offset;
        const int32x4_t voffset = vdupq_n_s32(offset);
        const float32x4_t vscale = vdupq_n_f32(scale);
        return vdequantize(qv, voffset, vscale);
    }

    /** Dequantize a neon vector holding 16 signed quantized values.
     *
     * @param[in] qv      Input values to be dequantized.
     * @param[in] voffset Vectorized dequantization offset.
     * @param[in] vscale  Vectorized dequantization scale.
     *
     * @return Dequantized values in a neon vector
     */
    inline float32x4x4_t vdequantize(const int8x16_t &qv, const int32x4_t &voffset, const float32x4_t &vscale) {
        const float32x4x4_t vdequantized_input = {
            {
                vmulq_f32(vcvtq_f32_s32(vsubq_s32(
                              vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(qv)))),
                              voffset)), vscale),
                vmulq_f32(vcvtq_f32_s32(vsubq_s32(
                              vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(qv)))),
                              voffset)), vscale),
                vmulq_f32(vcvtq_f32_s32(vsubq_s32(
                              vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(qv)))),
                              voffset)), vscale),
                vmulq_f32(vcvtq_f32_s32(vsubq_s32(
                              vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(qv)))),
                              voffset)), vscale),
            }
        };
        return vdequantized_input;
    }

    /** Dequantize a neon vector holding 16 signed quantized values.
     *
     * @param[in] qv Input values to be dequantized.
     * @param[in] qi Quantization information to be used in the computation.
     *
     * @return Dequantized values in a neon vector
     */
    inline float32x4x4_t vdequantize(const int8x16_t &qv, const BIUniformQuantizationInfo &qi) {
        const float scale = qi.scale;
        const int offset = qi.offset;
        const int32x4_t voffset = vdupq_n_s32(offset);
        const float32x4_t vscale = vdupq_n_f32(scale);
        return vdequantize(qv, voffset, vscale);
    }

    /** Dequantize following an asymmetric quantization scheme a neon vector holding 16 quantized values.
     *
     * @param[in] qv     Input values to be dequantized.
     * @param[in] scale  Quantization scaling factor.
     * @param[in] offset Zero quantization offset.
     *
     * @return Dequantized values in a neon vector
     */
    inline float32x4x4_t vdequantize(const uint8x16_t &qv, float scale, int32_t offset) {
        const int32x4_t voffset = vdupq_n_s32(offset);
        const float32x4_t vscale = vdupq_n_f32(scale);
        const float32x4x4_t vdequantized_input = {
            {
                vmulq_f32(vcvtq_f32_s32(
                              vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                                            vget_low_u16(vmovl_u8(vget_low_u8(qv))))),
                                        voffset)),
                          vscale),
                vmulq_f32(vcvtq_f32_s32(
                              vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                                            vget_high_u16(vmovl_u8(vget_low_u8(qv))))),
                                        voffset)),
                          vscale),
                vmulq_f32(vcvtq_f32_s32(
                              vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                                            vget_low_u16(vmovl_u8(vget_high_u8(qv))))),
                                        voffset)),
                          vscale),
                vmulq_f32(vcvtq_f32_s32(
                              vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                                            vget_high_u16(vmovl_u8(vget_high_u8(qv))))),
                                        voffset)),
                          vscale),
            }
        };
        return vdequantized_input;
    }

    /** Dequantize a vector of 16 values stored as signed asymmetric.
     *
     * @param[in] qv     Input values to be dequantized.
     * @param[in] scale  Quantization scaling factor.
     * @param[in] offset Zero quantization offset.
     *
     * @return Dequantized values in a neon vector
     */
    inline float32x4x4_t vdequantize(const int8x16_t &qv, float scale, int32_t offset) {
        const int32x4_t voffset = vdupq_n_s32(offset);
        const float32x4_t vscale = vdupq_n_f32(scale);
        const float32x4x4_t vdequantized_input = {
            {
                vmulq_f32(vcvtq_f32_s32(vsubq_s32(
                              vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(qv)))),
                              voffset)), vscale),
                vmulq_f32(vcvtq_f32_s32(vsubq_s32(
                              vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(qv)))),
                              voffset)), vscale),
                vmulq_f32(vcvtq_f32_s32(vsubq_s32(
                              vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(qv)))),
                              voffset)), vscale),
                vmulq_f32(vcvtq_f32_s32(vsubq_s32(
                              vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(qv)))),
                              voffset)), vscale),
            }
        };
        return vdequantized_input;
    }

    /** Dequantize following symmetric quantization scheme a neon vector holding 16 quantized values.
     *
     * @param[in] qv     Input values to be dequantized.
     * @param[in] vscale Vector containing quantization scaling factors.
     *
     * @return Dequantized values in a neon vector
     */
    inline float32x4x4_t vdequantize(const int8x16_t &qv, const float32x4x4_t vscale) {
        const float32x4x4_t vdequantized_input = {
            {
                vmulq_f32(vcvtq_f32_s32(
                              vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(qv))))),
                          vscale.val[0]),
                vmulq_f32(vcvtq_f32_s32(
                              vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(qv))))),
                          vscale.val[1]),
                vmulq_f32(vcvtq_f32_s32(
                              vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(qv))))),
                          vscale.val[2]),
                vmulq_f32(vcvtq_f32_s32(
                              vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(qv))))),
                          vscale.val[3]),
            }
        };
        return vdequantized_input;
    }

    /** Dequantize following a symmetric quantization scheme a neon vector holding 16 quantized values.
     *
     * @param[in] qv    Input values to be dequantized.
     * @param[in] scale Quantization scaling factor.
     *
     * @return Dequantized values in a neon vector
     */
    inline float32x4x4_t vdequantize(const int8x16_t &qv, float scale) {
        const float32x4_t vscale = vdupq_n_f32(scale);
        const float32x4x4_t vdequantized_input = {
            {
                vmulq_f32(vcvtq_f32_s32(
                              vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(qv))))),
                          vscale),
                vmulq_f32(vcvtq_f32_s32(
                              vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(qv))))),
                          vscale),
                vmulq_f32(vcvtq_f32_s32(
                              vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(qv))))),
                          vscale),
                vmulq_f32(vcvtq_f32_s32(
                              vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(qv))))),
                          vscale),
            }
        };
        return vdequantized_input;
    }

    /** Quantize a neon vector holding 8 floating point values.
     *
     * @param[in] qv Input values to be quantized.
     * @param[in] qi Quantization information to be used in the computation.
     *
     * @return A neon vector holding the quantized values
     */
    inline uint8x8_t vquantize(const float32x4x2_t &qv, const BIUniformQuantizationInfo &qi) {
        const float scale = qi.scale;
        const int offset = qi.offset;
        const float32x4_t voffset = vdupq_n_f32(offset);
        const float32x4_t vinvscale = vdupq_n_f32(1.f / scale);
        const int32x4x4_t rf = {
            {
#ifdef __aarch64__
                vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[0], vinvscale)),
                vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[1], vinvscale)),
#else  //__aarch64__
                                        vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[0], vinvscale)),
 vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[1], vinvscale)),
#endif //__aarch64__
            }
        };
        return vqmovun_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
    }

    /** Quantize a neon vector holding 8 floating point values.
     *
     * @param[in] qv Input values to be quantized.
     * @param[in] qi Quantization information to be used in the computation.
     *
     * @return A neon vector holding the singed quantized values
     */
    inline int8x8_t vquantize_signed(const float32x4x2_t &qv, const BIUniformQuantizationInfo &qi) {
        const float scale = qi.scale;
        const int offset = qi.offset;
        const float32x4_t voffset = vdupq_n_f32(offset);
        const float32x4_t vinvscale = vdupq_n_f32(1.f / scale);
        const int32x4x4_t rf = {
            {
#ifdef __aarch64__
                vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[0], vinvscale)),
                vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[1], vinvscale)),
#else  //__aarch64__
                                        vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[0], vinvscale)),
 vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[1], vinvscale)),
#endif //__aarch64__
            }
        };
        return vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
    }

    inline int32x4x4_t vquantize_internal(const float32x4x4_t &qv, float scale, int32_t offset) {
        const int32x4_t voffset = vdupq_n_s32(offset);
        const float32x4_t vinvscale = vdupq_n_f32(1.f / scale);
        const int32x4x4_t rf = {
            {
#ifdef __aarch64__
                vaddq_s32(vcvtnq_s32_f32(vmulq_f32(qv.val[0], vinvscale)), voffset),
                vaddq_s32(vcvtnq_s32_f32(vmulq_f32(qv.val[1], vinvscale)), voffset),
                vaddq_s32(vcvtnq_s32_f32(vmulq_f32(qv.val[2], vinvscale)), voffset),
                vaddq_s32(vcvtnq_s32_f32(vmulq_f32(qv.val[3], vinvscale)), voffset),
#else  //__aarch64__
                                        vaddq_s32(vcvtq_s32_f32(vmulq_f32(qv.val[0], vinvscale)), voffset),
 vaddq_s32(vcvtq_s32_f32(vmulq_f32(qv.val[1], vinvscale)), voffset),
 vaddq_s32(vcvtq_s32_f32(vmulq_f32(qv.val[2], vinvscale)), voffset),
 vaddq_s32(vcvtq_s32_f32(vmulq_f32(qv.val[3], vinvscale)), voffset),
#endif //__aarch64__
            }
        };
        return rf;
    }

    inline int32x4x4_t vquantize_internal(const float32x4x4_t &qv, float scale, float offset) {
        const float32x4_t voffset = vdupq_n_f32(offset);
        const float32x4_t vinvscale = vdupq_n_f32(1.f / scale);
        const int32x4x4_t rf = {
            {
#ifdef __aarch64__
                vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[0], vinvscale)),
                vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[1], vinvscale)),
                vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[2], vinvscale)),
                vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[3], vinvscale)),
#else  //__aarch64__
                                        vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[0], vinvscale)),
 vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[1], vinvscale)),
 vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[2], vinvscale)),
 vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[3], vinvscale)),
#endif //__aarch64__
            }
        };
        return rf;
    }

    /** Quantize a neon vector holding 16 floating point values.
     *
     * @param[in] qv Input values to be quantized.
     * @param[in] qi Quantization information to be used in the computation.
     *
     * @return A neon vector holding the quantized values
     */
    inline uint8x16_t vquantize(const float32x4x4_t &qv, const BIUniformQuantizationInfo &qi) {
        auto rf = vquantize_internal(qv, qi.scale, qi.offset);
        const uint8x8_t pa = vqmovun_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
        const uint8x8_t pb = vqmovun_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
        return vcombine_u8(pa, pb);
    }

    inline uint8x16_t vquantize(const float32x4x4_t &qv, const BIUniformRequantizationInfo &qi) {
        auto rf = vquantize_internal(qv, qi.scale, qi.offset);
        const uint8x8_t pa = vqmovun_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
        const uint8x8_t pb = vqmovun_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
        return vcombine_u8(pa, pb);
    }

    /** Signed quantize a neon vector holding 16 floating point values.
     *
     * @param[in] qv Input values to be quantized.
     * @param[in] qi Quantization information to be used in the computation.
     *
     * @return A neon vector holding the quantized values
     */
    inline int8x16_t vquantize_signed(const float32x4x4_t &qv, const BIUniformQuantizationInfo &qi) {
        auto rf = vquantize_internal(qv, qi.scale, qi.offset);
        const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
        const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
        return vcombine_s8(pa, pb);
    }

    inline int8x16_t vquantize_signed(const float32x4x4_t &qv, const BIUniformRequantizationInfo &qi) {
        auto rf = vquantize_internal(qv, qi.scale, qi.offset);
        const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
        const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
        return vcombine_s8(pa, pb);
    }

    /** Quantize to QASYMM16 a neon vector holding 16 floating point values.
     *
     * @param[in] qv Input values to be quantized.
     * @param[in] qi Quantization information to be used in the computation.
     *
     * @return A neon vector holding the quantized values
     */
    inline uint16x8x2_t vquantize_qasymm16(const float32x4x4_t &qv, const BIUniformQuantizationInfo &qi) {
        auto rf = vquantize_internal(qv, qi.scale, qi.offset);
        const uint16x8_t pa = vcombine_u16(vqmovun_s32(rf.val[0]), vqmovun_s32(rf.val[1]));
        const uint16x8_t pb = vcombine_u16(vqmovun_s32(rf.val[2]), vqmovun_s32(rf.val[3]));
        return {pa, pb};
    }
}

#include <data/core/neon/bi_ne_asymm.inl>

#endif //BATMANINFER_BI_NE_ASYMM_HPP
