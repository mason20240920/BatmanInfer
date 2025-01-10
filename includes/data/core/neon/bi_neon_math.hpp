//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_BI_NEON_MATH_HPP
#define BATMANINFER_BI_NEON_MATH_HPP

#include <data/core/bi_rounding.h>

#include <neon/neon_defines.h>
#include <array>

namespace BatmanInfer {
    /**
     * float32x4_t: 占用 128 位（16 字节）的存储空间
     * 加法：vaddq_f32（对两个 float32x4_t 向量的每个元素执行加法）
     * 乘法：vmulq_f32（对两个 float32x4_t 向量的每个元素执行乘法）
     * 负数：vnegq_f32（对每个元素取负值）
     *
     * @desc 计算向量的下取整。
     *
     * @return 向下取整的向量
     *
     */
    float32x4_t vfloorq_f32(float32x4_t val);


    /**
     * 计算向量到最近的四舍五入值，当相邻时取偶数。
     *
     * | 原始值	| 四舍五入结果（取偶数）|
     * | 2.3	|  2                |
     * | 2.5	| 2 （取偶数）
     * | 3.5	| 4 （取偶数）
     * | 4.7	| 5
     *
     * @param val
     * @return
     */
    float32x4_t vroundq_rte_f32(float32x4_t val);

    /**
     *
     * @param x
     * @return
     */
    float32x2_t vinvsqrt_f32(float32x2_t x);

/** Calculate inverse square root.
 *
 * @param[in] x Input value.
 *
 * @return The calculated inverse square root.
 */
    float32x4_t vinvsqrtq_f32(float32x4_t x);

/** Calculate reciprocal.
 *
 * @param[in] x Input value.
 *
 * @return The calculated reciprocal.
 */
    float32x2_t vinv_f32(float32x2_t x);

/** Calculate reciprocal.
 *
 * @param[in] x Input value.
 *
 * @return The calculated reciprocal.
 */
    float32x4_t vinvq_f32(float32x4_t x);

    /** Perform a 7th degree polynomial approximation using Estrin's method.
     *
     * @param[in] x      Input vector value in F32 format.
     * @param[in] coeffs Polynomial coefficients table.
     *
     * @return The calculated approximation.
     */
    float32x4_t vtaylor_polyq_f32(float32x4_t x, const std::array<float32x4_t, 8> &coeffs);

/** Calculate exponential
 *
 * @param[in] x Input vector value in F32 format.
 *
 * @return The calculated exponent.
 */
    float32x4_t vexpq_f32(float32x4_t x);

/** Calculate error function
 *
 * @param[in] x Input vector in F32 format.
 *
 * @return The calculated erf.
 */
    float32x4_t verfq_f32(float32x4_t x);

/** Calculate logarithm
 *
 * @param[in] x Input vector value in F32 format.
 *
 * @return The calculated logarithm.
 */
    float32x4_t vlogq_f32(float32x4_t x);

/** Calculate hyperbolic tangent.
 *
 * tanh(x) = (e^2x - 1)/(e^2x + 1)
 *
 * @note We clamp x to [-5,5] to avoid overflowing issues.
 *
 * @param[in] val Input vector value in F32 format.
 *
 * @return The calculated Hyperbolic Tangent.
 */
    float32x4_t vtanhq_f32(float32x4_t val);

/** Calculate n power of a number.
 *
 * pow(x,n) = e^(n*log(x))
 *
 * @param[in] val Input vector value in F32 format.
 * @param[in] n   Powers to raise the input to.
 *
 * @return The calculated power.
 */
    float32x4_t vpowq_f32(float32x4_t val, float32x4_t n);

/** Round to the nearest division by a power-of-two using exponent
 *
 * @note This function calculates the following expression: (x + 2^n -1 ) / 2^n where n = exponent
 *
 * @param[in] x        Vector of 4 elements
 * @param[in] exponent Vector of 4 elements with integer value used to round to nearest division by a power-of-two
 *
 * @return the nearest division by a power-of-two using exponent
 */
    int32x4_t rounding_divide_by_pow2(int32x4_t x, int32x4_t exponent);

/** Round to the nearest division by a power-of-two using exponent
 *
 * @note This function calculates the following expression: (x + 2^n -1 ) / 2^n where n = exponent
 *
 * @param[in] x        Vector of 4 elements
 * @param[in] exponent Integer value used to round to nearest division by a power-of-two
 *
 * @return the nearest division by a power-of-two using exponent
 */
    int32x4_t rounding_divide_by_pow2(int32x4_t x, int exponent);

/** Round to the nearest division by a power-of-two using exponent
 *
 * @note This function calculates the following expression: (x + 2^n -1 ) / 2^n where n = exponent
 *
 * @param[in] x        Element to divide.
 * @param[in] exponent Integer value used to round to nearest division by a power-of-two
 *
 * @return the nearest division by a power-of-two using exponent
 */
    int32_t rounding_divide_by_pow2(int32_t x, int exponent);

/** Converts from uint8x16 to float32x4x4_t
 *
 * @param[in] in Vector of uint8 to be converted
 *
 * @return Converted vector of float
 */
    float32x4x4_t convert_uint8x16_to_float32x4x4(const uint8x16_t &in);

/** Converts from int8x16 to float32x4x4_t
 *
 * @param[in] in Vector of int8 to be converted
 *
 * @return Converted vector of float
 */
    float32x4x4_t convert_int8x16_to_float32x4x4(const int8x16_t &in);

/** Converts to float32x4x4_t from the specified templated 16 elements vectors
 *
 * @param[in] in Vector of float to be converted
 *
 * @return Converted vector of float
 */
    template<typename T>
    float32x4x4_t convert_to_float32x4x4(const T &in);

/** Converts from two float32x4x3_t to just one uint8x8x3_t
 *
 * @param[in]  in1 First input vector of float to be converted
 * @param[in]  in2 Second input vector of float to be converted
 * @param[out] out Converted output vector uint8 to store the result
 */
    void convert_float32x4x3_to_uint8x8x3(const float32x4x3_t &in1, const float32x4x3_t &in2, uint8x8x3_t &out);

/** Converts from two float32x4x4_t to just one uint8x16_t
 *
 * @param[in]  in  Vector of float to be converted
 * @param[out] out Converted vector of uint8 to store the result
 */
    template<BIRoundingPolicy policy = BIRoundingPolicy::TO_ZERO>
    void convert_float32x4x4_to_uint8x16(const float32x4x4_t &in, uint8x16_t &out);

/** Converts from float32x4x4_t to just one int8x16_t
 *
 * @param[in]  in  Vector of float to be converted
 * @param[out] out Converted vector of uint8 to store the result
 */
    template<BIRoundingPolicy policy = BIRoundingPolicy::TO_ZERO>
    void convert_float32x4x4_to_int8x16(const float32x4x4_t &in, int8x16_t &out);

/** Converts from float vector to integer vector
 *
 * @note: Default rounding mode is "Round to Nearest with Ties to Even"
 *        if __aarch64__ is defined else "Round towards Zero"
 *
 * @param[in] in Float vector to converted
 *
 * @return The converted integer vector
 */
    template<typename float_vec_type, typename int_vec_type>
    int_vec_type convert_float_to_int(const float_vec_type &in);

/** Converts from integer vector to float vector
 *
 * @param[in] in Integer vector to converted
 *
 * @return The converted float vector
 */
    template<typename float_vec_type, typename int_vec_type>
    float_vec_type convert_int_to_float(const int_vec_type &in);

/** Calculate sine.
 *
 * @param[in] val Input vector value in radians, F32 format.
 *
 * @return The calculated sine.
 */
    float32x4_t vsinq_f32(float32x4_t val);

/** Calculate sine.
 *
 * @param[in] val Input vector value in radians, F32 format.
 *
 * @return The calculated sine.
 */
    float32x2_t vsin_f32(float32x2_t val);

/** Reduce a vector to be a scalar by accumulating all lanes in the vector
 *
 * @param[in] v Vector to be reduced.
 *
 * @return the wrapped-around number.
 */
    float vreduce(const float32x4_t &v);

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

/** Calculate hyperbolic tangent.
 *
 * tanh(x) = (e^2x - 1)/(e^2x + 1)
 *
 * @note We clamp x to [-5,5] to avoid overflowing issues.
 *
 * @param[in] val Input vector value in F16 format.
 *
 * @return The calculated Hyperbolic Tangent.
 */
    float16x8_t vtanhq_f16(float16x8_t val);

/** Calculate round value of a vector to nearest with ties to even.
 *
 * @param[in] val Input vector value in F16 format.
 *
 * @return The calculated round vector.
 */
    float16x8_t vroundq_rte_f16(float16x8_t val);

/** Calculate reciprocal.
 *
 * @param[in] x Input value.
 *
 * @return The calculated reciprocal.
 */
    float16x4_t vinv_f16(float16x4_t x);

/** Calculate reciprocal.
 *
 * @param[in] x Input value.
 *
 * @return The calculated reciprocal.
 */
    float16x8_t vinvq_f16(float16x8_t x);

/** Calculate inverse square root.
 *
 * @param[in] x Input value.
 *
 * @return The calculated inverse square root.
 */
    float16x4_t vinvsqrt_f16(float16x4_t x);

/** Calculate inverse square root.
 *
 * @param[in] x Input value.
 *
 * @return The calculated inverse square root.
 */
    float16x8_t vinvsqrtq_f16(float16x8_t x);

/** Calculate exponential
 *
 * @param[in] x Input vector value in F16 format.
 *
 * @return The calculated exponent.
 */
    float16x8_t vexpq_f16(float16x8_t x);

/** Calculate error function
 *
 * @param[in] x Input vector in F16 format.
 *
 * @return The calculated erf.
 */
    float16x8_t verfq_f16(float16x8_t x);

/** Calculate n power of a number.
 *
 * pow(x,n) = e^(n*log(x))
 *
 * @param[in] val Input vector value in F16 format.
 * @param[in] n   Powers to raise the input to.
 *
 * @return The calculated power.
 */
    float16x8_t vpowq_f16(float16x8_t val, float16x8_t n);

/** Calculate sine.
 *
 * @param[in] val Input vector value in radians, F16 format.
 *
 * @return The calculated sine.
 */
    float16x8_t vsinq_f16(float16x8_t val);

/** Reduce a vector to be a scalar by accumulating all lanes in the vector
 *
 * @param[in] v Vector to be reduced.
 *
 * @return the wrapped-around number.
 */
    float16_t vreduce(const float16x8_t &v);

#endif
}

#include <data/core/neon/bi_neon_math.inl>

#endif //BATMANINFER_BI_NEON_MATH_HPP
