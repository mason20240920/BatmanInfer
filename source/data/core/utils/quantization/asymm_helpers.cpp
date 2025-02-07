//
// Created by Mason on 2025/1/10.
//

#include <data/core/utils/quantization/asymm_helpers.hpp>

#include <data/core/bi_helpers.hpp>
#include <function_info/bi_activationLayerInfo.h>

#include <support/bi_toolchain_support.hpp>

#include <cmath>
#include <limits>

namespace BatmanInfer {
    namespace quantization {
        constexpr int64_t fixed_point_one_Q0 = (1LL << 31);
        constexpr float epsilon = 0.00001f;

        BIStatus calculate_quantized_multiplier(float multiplier, int32_t *quant_multiplier, int32_t *shift,
                                                bool ignore_epsilon) {
            if (multiplier >= 1.f) {
                BIStatus status = calculate_quantized_multiplier_greater_than_one(multiplier, quant_multiplier, shift);
                *shift *= -1;
                return status;
            } else {
                return calculate_quantized_multiplier_less_than_one(multiplier, quant_multiplier, shift,
                                                                    ignore_epsilon);
            }
        }

        BIStatus calculate_quantized_multiplier_less_than_one(float multiplier,
                                                              int32_t *quant_multiplier,
                                                              int32_t *right_shift,
                                                              bool ignore_epsilon) {
            // 根据 ignore_epsilon 参数设置内部的 epsilon 值，
            // 如果忽略 epsilon，则内部容差设置为 0；否则为预先定义的 epsilon。
            const float internal_epsilon = ignore_epsilon ? 0.0f : epsilon;

            // 对输入指针进行空指针检查，确保 quant_multiplier 与 right_shift 非空。
            BI_COMPUTE_RETURN_ERROR_ON(quant_multiplier == nullptr);
            BI_COMPUTE_RETURN_ERROR_ON(right_shift == nullptr);

            // 检查 multiplier 是否小于 -epsilon 或大于 1 + epsilon，
            // 确保输入乘子在合理的范围内（乘子一般应该小于等于 1）。
            BI_COMPUTE_RETURN_ERROR_ON(multiplier < -internal_epsilon);
            BI_COMPUTE_RETURN_ERROR_ON(multiplier > 1.0f + internal_epsilon);

            int shift_exp = 0;
            // std::frexp 将 multiplier 表示为 q * 2^(shift_exp) 的形式，其中 q 在 [0.5, 1) 内
            const double q = std::frexp(multiplier, &shift_exp);
            // 右移位数取 -shift_exp，因为后续在定点数中需要进行右移操作（负的 shift_exp 意味着乘上 2^(|shift_exp|)）
            *right_shift = -1 * shift_exp;

            // 将 q 转换为定点表示。fixed_point_one_Q0 表示一个定点数常量（例如 2^n 的值），
            // 这里通过四舍五入将 q 转换为整数形式。
            auto q_fixed = static_cast<int64_t>(support::cpp11::round(q * fixed_point_one_Q0));
            // 检查 q_fixed 是否超过了 fixed_point_one_Q0，确保结果不会溢出（理论上最多等于 fixed_point_one_Q0）。
            BI_COMPUTE_RETURN_ERROR_ON(q_fixed > fixed_point_one_Q0);

            // 如果 q_fixed 恰好等于 fixed_point_one_Q0，则将其调整为一半，同时将右移位数减 1，
            // 目的是保证 q_fixed 落在期望的范围内（例如防止乘子溢出）。
            if (q_fixed == fixed_point_one_Q0) {
                q_fixed /= 2;
                --*right_shift;
            }

            // 当忽略 epsilon 而且右移位数过大（大于 31）时，将右移位数设置为 0，并置 q_fixed 为 0，
            // 这种情况表示乘子非常小，从定点数角度看可以直接归零。
            if (ignore_epsilon && *right_shift > 31) {
                *right_shift = 0;
                q_fixed = 0;
            }

            // 检查右移位数是否为负值（负值不符合逻辑，应始终为非负数）。
            BI_COMPUTE_RETURN_ERROR_ON(*right_shift < 0);
            // 检查 q_fixed 是否超过 int32_t 的最大值，确保定点表示能够正确存储在 32 位整数中。
            BI_COMPUTE_RETURN_ERROR_ON(q_fixed > std::numeric_limits<int32_t>::max());

            // 最后，将计算得到的定点 q_fixed 赋值到输出变量 quant_multiplier 中。
            *quant_multiplier = static_cast<int32_t>(q_fixed);

            return BIStatus{};
        }

        BIStatus
        calculate_quantized_multiplier_greater_than_one(float multiplier, int32_t *quantized_multiplier,
                                                        int32_t *left_shift) {
            BI_COMPUTE_RETURN_ERROR_ON(quantized_multiplier == nullptr);
            BI_COMPUTE_RETURN_ERROR_ON(left_shift == nullptr);
            BI_COMPUTE_RETURN_ERROR_ON(multiplier < 1.f);

            int shift_exp = 0;
            const double q = std::frexp(multiplier, &shift_exp);
            *left_shift = shift_exp;
            auto q_fixed = static_cast<int64_t>(support::cpp11::round(q * fixed_point_one_Q0));
            BI_COMPUTE_RETURN_ERROR_ON(q_fixed > fixed_point_one_Q0);
            if (q_fixed == fixed_point_one_Q0) {
                q_fixed /= 2;
                ++*left_shift;
            }
            BI_COMPUTE_RETURN_ERROR_ON(*left_shift < 0);
            BI_COMPUTE_RETURN_ERROR_ON(q_fixed > std::numeric_limits<int32_t>::max());
            *quantized_multiplier = static_cast<int32_t>(q_fixed);

            return BIStatus{};
        }

        BatmanInfer::BIStatus calculate_quantized_multipliers(const BIQuantizationInfo &iq_info,
                                                              const BIQuantizationInfo &wq_info,
                                                              const BIQuantizationInfo &oq_info,
                                                              BIGEMMLowpOutputStageInfo &stage_info) {
            BI_COMPUTE_RETURN_ERROR_ON(iq_info.scale().empty());
            BI_COMPUTE_RETURN_ERROR_ON(wq_info.scale().empty());
            BI_COMPUTE_RETURN_ERROR_ON(oq_info.scale().empty());
            constexpr unsigned int padding_elems = 32; // assembly kernels assume the shifts and multipliers buffers are padded
            const unsigned int size = wq_info.scale().size();
            const size_t padded_size = (size == 1) ? 1 : size + padding_elems;
            auto &quant_multipliers = stage_info.gemmlowp_multipliers;
            auto &quant_shifts = stage_info.gemmlowp_shifts;
            quant_multipliers.resize(padded_size);
            quant_shifts.resize(padded_size);

            const auto &w_scales = wq_info.scale();
            const float i_scale = iq_info.scale().at(0);
            const float o_scale = oq_info.scale().at(0);

            for (unsigned int i = 0; i < size; ++i) {
                const float multiplier = i_scale * w_scales[i] / o_scale;
                int32_t quant_multiplier = 0;
                int32_t quant_shift = 0;
                BI_COMPUTE_RETURN_ON_ERROR(
                        calculate_quantized_multiplier(multiplier, &quant_multiplier, &quant_shift));
                quant_multipliers[i] = quant_multiplier;
                quant_shifts[i] = quant_shift;
            }

            // Legacy part
            stage_info.gemmlowp_shift = quant_shifts[0];
            stage_info.gemmlowp_multiplier = quant_multipliers[0];

            return BIStatus{};
        }

        std::pair<int, int> get_min_max_values_from_quantized_data_type(BIDataType data_type) {
            int min_quant_val = 0;
            int max_quant_val = 0;
            switch (data_type) {
                case BIDataType::QASYMM8:
                    min_quant_val = std::numeric_limits<uint8_t>::min();
                    max_quant_val = std::numeric_limits<uint8_t>::max();
                    break;
                case BIDataType::QSYMM8:
                case BIDataType::QASYMM8_SIGNED:
                    min_quant_val = std::numeric_limits<int8_t>::min();
                    max_quant_val = std::numeric_limits<int8_t>::max();
                    break;
                case BIDataType::QASYMM16:
                    min_quant_val = std::numeric_limits<uint16_t>::min();
                    max_quant_val = std::numeric_limits<uint16_t>::max();
                    break;
                case BIDataType::QSYMM16:
                    min_quant_val = std::numeric_limits<int16_t>::min();
                    max_quant_val = std::numeric_limits<int16_t>::max();
                    break;
                default:
                    BI_COMPUTE_ERROR("Unsupported data type");
            }
            return std::make_pair(min_quant_val, max_quant_val);
        }

        std::tuple<int32_t, int32_t> get_quantized_asymmetric_output_min_max(const BIQuantizationInfo &q_info,
                                                                             const BIActivationLayerInfo &act_info,
                                                                             BIDataType data_type) {
            BI_COMPUTE_ERROR_ON(data_type != BIDataType::QASYMM8 && data_type != BIDataType::QASYMM8_SIGNED);

            const auto min_max = get_min_max(data_type);

            int32_t type_min = std::get<0>(min_max).get<int32_t>();
            int32_t type_max = std::get<1>(min_max).get<int32_t>();

            const BIUniformQuantizationInfo q_unif = q_info.uniform();

            if (act_info.enabled()) {
                switch (act_info.activation()) {
                    case BIActivationLayerInfo::ActivationFunction::RELU:
                        type_min = q_unif.offset;
                        break;
                    case BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                        type_min = q_unif.offset;
                        type_max = (data_type == BIDataType::QASYMM8) ? quantize_qasymm8(act_info.a(), q_info)
                                                                      : quantize_qasymm8_signed(act_info.a(), q_info);
                        break;
                    case BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
                        type_min = (data_type == BIDataType::QASYMM8) ? quantize_qasymm8(act_info.b(), q_info)
                                                                      : quantize_qasymm8_signed(act_info.b(), q_info);
                        type_max = (data_type == BIDataType::QASYMM8) ? quantize_qasymm8(act_info.a(), q_info)
                                                                      : quantize_qasymm8_signed(act_info.a(), q_info);
                        break;
                    default:
                        BI_COMPUTE_ERROR("Activation function not supported.");
                        break;
                }
            }

            return std::make_tuple(type_min, type_max);
        }

        void compute_quantized_multipliers_and_shifts(const BIITensorInfo *input,
                                                      const BIITensorInfo *weights,
                                                      const BIITensorInfo *output,
                                                      int32_t *output_multipliers_ptr,
                                                      int32_t *output_shifts_ptr) {
            const BIUniformQuantizationInfo iq_info = input->quantization_info().uniform();
            const BIQuantizationInfo wq_info = weights->quantization_info();
            const BIUniformQuantizationInfo oq_info = output->quantization_info().uniform();

            const unsigned int num_filters = wq_info.scale().size();

            for (unsigned int i = 0; i < num_filters; ++i) {
                int32_t output_multiplier = 0;
                int32_t output_shift = 0;
                const float multiplier = iq_info.scale * wq_info.scale()[i] / oq_info.scale;
                calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);

                output_multipliers_ptr[i] = output_multiplier;
                output_shifts_ptr[i] = output_shift;
            }
        }

        int32_t saturating_rounding_doubling_highmul(int32_t a, int32_t b) {
            bool overflow = a == b && a == std::numeric_limits<int32_t>::min();
            int64_t a_64(a);
            int64_t b_64(b);
            int64_t ab_x2_64 = a_64 * b_64 * 2;
            return overflow ? std::numeric_limits<int32_t>::max() : (ab_x2_64 >> 32);
        }

        inline int32_t rounding_divide_by_pow2(int32_t x, int exponent) {
            return (exponent == 0) ? x : ((x + (1 << (exponent - 1))) >> exponent);
        }

        int32_t multiply_by_quantized_multiplier(int32_t input, int32_t qmul, int32_t shift) {
            const auto left_shift = shift > 0 ? shift : 0;
            const auto right_shift = shift > 0 ? 0 : -shift;
            return rounding_divide_by_pow2(saturating_rounding_doubling_highmul(input * (1 << left_shift), qmul),
                                           right_shift);
        }

        int32_t saturating_rounding_multiply_by_pow2(int32_t exponent, int32_t v) {
            if (exponent == 0) {
                return v;
            } else if (exponent < 0) {
                return rounding_divide_by_pow2(v, -exponent);
            } else {
                constexpr auto min = std::numeric_limits<int32_t>::min();
                constexpr auto max = std::numeric_limits<int32_t>::max();
                const auto width = sizeof(int32_t) * 8;

                const int32_t threshold = ((1 << (width - 1 - exponent)) - 1);
                bool pos_mask = v > threshold;
                bool neg_mask = v < -threshold;
                int32_t result = v << exponent;
                result = pos_mask ? max : result;
                result = neg_mask ? min : result;
                return result;
            }
        }

        void get_invsqrt_quantized_multiplier_exp(int32_t input,
                                                  int32_t reverse_shift,
                                                  int32_t &output_inv_sqrt,
                                                  int32_t &output_shift) {
            BI_COMPUTE_ERROR_ON(input < 0);

            if (input <= 1) {
                // dealing the inputs (0 and 1) separately to avoid overflow
                output_inv_sqrt = std::numeric_limits<std::int32_t>::max();
                output_shift = 0;
                return;
            }

            // prepare input for fixed point operation and compute shift value
            output_shift = 11;
            while (input >= (1 << 29)) {
                input /= 4;
                ++output_shift;
            }

            const uint32_t max_left_shift_bits = __builtin_clz(static_cast<uint32_t>(input)) - 1;
            const uint32_t max_left_shift_bits_pairs = max_left_shift_bits / 2;
            const uint32_t left_shift_bit_pairs = max_left_shift_bits_pairs - 1;
            output_shift -= left_shift_bit_pairs;
            input <<= 2 * left_shift_bit_pairs;

            // Calculation in fixed point domain with 3 integer bits.
            using FixedPointRawType = int32_t;
            constexpr uint32_t fixedpoint_position = 3;
            constexpr uint32_t fixedpoint_int_position = sizeof(FixedPointRawType) * 8 - 1 - fixedpoint_position;
            using FixedPoint3 = FixedPointRawType;
            using FixedPoint0 = FixedPointRawType;

            // fixed point representation of input divided by 2 and 1.5 for Newton-Raphson iteration
            const FixedPoint3 fixedpoint_input = (input >> 1);
            const FixedPoint3 fixedpoint_half_input = rounding_divide_by_pow2(fixedpoint_input, 1);
            const FixedPoint3 fixedpoint_half_three =
                    (0x1 << fixedpoint_int_position) + (0x1 << (fixedpoint_int_position - 1));

            // initial guess (1) in fixed point representation
            FixedPoint3 x = 0x1 << fixedpoint_int_position;

            // multiplication of two fixed point numbers, defined for readability
            auto fixed_point_mul = [](FixedPointRawType a, FixedPointRawType b) -> FixedPointRawType {
                return saturating_rounding_doubling_highmul(a, b);
            };

            // rescaling of fixed point to have dst_bit integer bits, defined for readability
            auto fixed_point_rescale = [](FixedPointRawType a, uint32_t src_bit,
                                          uint32_t dst_bit) -> FixedPointRawType {
                const uint32_t exponent = src_bit - dst_bit;
                return saturating_rounding_multiply_by_pow2(exponent, a);
            };

            // 5 iterations of Newton-Raphson method for inverse square root - 1.5 * x_n = input/2 * (x_n)^3
            constexpr int32_t num_iteration = 5;
            for (int32_t i = 0; i < num_iteration; ++i) {
                const auto x3 = fixed_point_rescale(fixed_point_mul(fixed_point_mul(x, x), x), 9, fixedpoint_position);
                x = fixed_point_rescale(
                        fixed_point_mul(fixedpoint_half_three, x) - fixed_point_mul(fixedpoint_half_input, x3),
                        6, fixedpoint_position);
            }

            // fixed point representation of sqrt(1/2)
            const FixedPoint0 fixedpoint_half_sqrt_2 = 1518500250;
            x = fixed_point_mul(fixedpoint_half_sqrt_2, x);
            output_inv_sqrt = x;
            if (output_shift < 0) {
                output_inv_sqrt <<= -output_shift;
                output_shift = 0;
            }
            // convert right shift to left shift
            output_shift *= reverse_shift;
        }
    }
}