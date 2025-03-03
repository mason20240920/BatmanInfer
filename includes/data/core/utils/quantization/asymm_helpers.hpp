//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_QUANTIZATION_ASYMM_HELPERS_HPP
#define BATMANINFER_QUANTIZATION_ASYMM_HELPERS_HPP

#include <data/core/bi_error.h>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>

#include <function_info/bi_GEMMInfo.h>

namespace BatmanInfer {
    namespace quantization {
        /** Calculate quantized representation of multiplier.
 *
 * @param[in]  multiplier       Real multiplier.
 * @param[out] quant_multiplier Integer multiplier.
 * @param[out] shift            bit shift. A negative value indicates a left shift, while a positive value indicates a right shift
 * @param[in]  ignore_epsilon   When true, ignore pre-defined epsilon value. Defaults to false
 *
 * @return a status
 */
        BIStatus calculate_quantized_multiplier(float multiplier,
                                                int32_t *quant_multiplier,
                                                int32_t *shift,
                                                bool ignore_epsilon = false);


        /** 计算小于 1 的量化乘子参数
         *
         * @param[in]  multiplier       Real multiplier. (输入的浮点乘子，要求在 [-epsilon, 1 + epsilon] 范围内，其中 epsilon 用于数值容差。)
         * @param[out] quant_multiplier Integer multiplier. (输出的量化乘子（整数），用于后续的定点运算)
         * @param[out] right_shift      Right bit shift. (输出的右移位数，用于对乘子进行缩放调整（右移表示除以 2^n）)
         * @param[in]  ignore_epsilon   When true, ignore pre-defined epsilon value. Defaults to false (是否忽略 epsilon 容差的检查（在某些情况中可能不考虑数值容差）。)
         *
         * @return a status
         */
        BIStatus calculate_quantized_multiplier_less_than_one(float multiplier,
                                                              int32_t *quant_multiplier,
                                                              int32_t *right_shift,
                                                              bool ignore_epsilon = false);

/** Calculate quantized representation of multiplier having value greater than one.
 *
 * @param[in]  multiplier           Real multiplier.
 * @param[out] quantized_multiplier Integer multiplier.
 * @param[out] left_shift           Left bit shift.
 *
 * @return a status
 */
        BIStatus
        calculate_quantized_multiplier_greater_than_one(float multiplier, int32_t *quantized_multiplier,
                                                        int32_t *left_shift);

/** Calculate quantized representation of per-channel multipliers
 *
 * @param[in]      iq_info    Input quantization info.
 * @param[in]      wq_info    Weights quantization info.
 * @param[in]      oq_info    Output quantization info.
 * @param[in, out] stage_info GemmLowp output stage info
 *
 * @return a status
 */
        BIStatus calculate_quantized_multipliers(const BIQuantizationInfo &iq_info,
                                                 const BIQuantizationInfo &wq_info,
                                                 const BIQuantizationInfo &oq_info,
                                                 BIGEMMLowpOutputStageInfo &stage_info);

/** Get minimum and maximum values for the input quantized data type
 *
 * @return min and max values for the quantized data type
 */
        std::pair<int, int> get_min_max_values_from_quantized_data_type(BIDataType data_type);

/** Compute quantized per-channel multipliers and shifts. As many multipliers
 * and shifts as output channels are computed. If weights are not quantized
 * per-channel, multipliers and shifts will end up being the same for each
 * channel.
 *
 * @param[in]  input                  Input tensor info.
 * @param[in]  weights                Weights tensor info.
 * @param[in]  output                 Output tensor info.
 * @param[out] output_multipliers_ptr Pointer to the buffer where to store per-channel multipliers.
 * @param[out] output_shifts_ptr      Pointer to the buffer where to store per-channel shifts.
 */
        void compute_quantized_multipliers_and_shifts(const BIITensorInfo *input,
                                                      const BIITensorInfo *weights,
                                                      const BIITensorInfo *output,
                                                      int32_t *output_multipliers_ptr,
                                                      int32_t *output_shifts_ptr);

/** Round to the nearest division by a power-of-two using exponent, copied from NEMath
 *
 * @note This function calculates the following expression: (x + 2^n -1 ) / 2^n where n = exponent
 *
 * @param[in] x        Element to divide.
 * @param[in] exponent Integer value used to round to nearest division by a power-of-two
 *
 * @return the nearest division by a power-of-two using exponent
 */
        int32_t rounding_divide_by_pow2(int32_t x, int exponent);

/** Compute multiplication of two integers
 *
 * @param[in] a One integer to multiply
 * @param[in] b Another integer to multiply
 *
 * @return The multiplied value
 */
        int32_t saturating_rounding_doubling_highmul(int32_t a, int32_t b);

/** Compute the value multiplied by given quantized multiplier and shift
 *
 * @param[in] input Target value to multiply.
 * @param[in] qmul  Quantized multipler
 * @param[in] shift Left bit shift
 *
 * @return The multiplied value
 */
        int32_t multiply_by_quantized_multiplier(int32_t input, int32_t qmul, int32_t shift);

/** Compute the value multiplied the power-of-two
 *
 * @param[in] exponent Exponent used to calculate power-of-two
 * @param[in] v        Target value to multiply
 *
 * @return The multiplied value
 */
        int32_t saturating_rounding_multiply_by_pow2(int32_t exponent, int32_t v);

/** Compute quantized multiplier and shift for the inverse square root of input.
 *  Using 3-bit fixed point and 5 iteration of Newton-Raphson method.
 *
 * @param[in]  input           Input to use
 * @param[in]  reverse_shift   -1 to reverse the shift direction
 * @param[out] output_inv_sqrt Quantized multiplier for inverse square root
 * @param[out] output_shift    Shift for inverse square root
 *
 */
        void get_invsqrt_quantized_multiplier_exp(int32_t input,
                                                  int32_t reverse_shift,
                                                  int32_t &output_inv_sqrt,
                                                  int32_t &output_shift);

        /** Get minimum and maximum output of the activation function after quantization.
 *
 * Only ReLU, upper bounded ReLU and lower+upper bounded ReLU are supported.
 *
 * @param[in] q_info    Output quantization info.
 * @param[in] act_info  Activation function information.
 * @param[in] data_type Output data type (either QASYMM8 or QASYMM8_SIGNED).
 *
 * @return The minimum and maximum output of the activation function after quantization.
 */
        std::tuple<int32_t, int32_t> get_quantized_asymmetric_output_min_max(const BIQuantizationInfo &q_info,
                                                                             const BIActivationLayerInfo &act_info,
                                                                             BIDataType data_type);
    }
}

#endif //BATMANINFER_QUANTIZATION_ASYMM_HELPERS_HPP
