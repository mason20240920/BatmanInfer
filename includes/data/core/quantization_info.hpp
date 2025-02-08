//
// Created by Mason on 2024/12/25.
//

#ifndef BATMANINFER_QUANTIZATION_INFO_HPP
#define BATMANINFER_QUANTIZATION_INFO_HPP

#include "data/core/bi_rounding.h"
#include "support/bi_toolchain_support.hpp"
#include <data/core/utils/misc/utils.hpp>

#include <vector>
#include <cstdint>

namespace BatmanInfer {

    using qasymm8_signed_t = int8_t;   /**< 8 bit signed quantized asymmetric scalar value */
    using qasymm8_t = uint8_t;  /**< 8 bit quantized asymmetric scalar value */
    using qsymm16_t = int16_t;  /**< 16 bit quantized symmetric scalar value */
    using qasymm16_t = uint16_t; /**< 16 bit quantized asymmetric scalar value */

    /** Quantization info when assuming per layer quantization */
    struct BIUniformQuantizationInfo {
        /** Default constructor */
        BIUniformQuantizationInfo() : scale(0.f), offset(0) {
        }

        /** Constructor
         *
         * @param[in] scale  Quantization scale
         * @param[in] offset Quantization offset
         */
        BIUniformQuantizationInfo(float scale, int32_t offset) : scale(scale), offset(offset) {
        }

        /** Checks if the scale and offset are both zero */
        bool empty() const {
            return (scale == 0) && (offset == 0);
        }

        float scale;
        int32_t offset;
    };

    /** Quantization info when assuming per layer quantization */
    struct BIUniformRequantizationInfo {
        /** Default constructor */
        BIUniformRequantizationInfo() : scale(0.f), offset(0.f) {
        }

        /** Constructor
         *
         * @param[in] scale  Quantization scale
         * @param[in] offset Quantization offset
         */
        BIUniformRequantizationInfo(float scale, float offset) : scale(scale), offset(offset) {
        }

        /** Checks if the scale and offset are both zero */
        bool empty() const {
            return (scale == 0) && (offset == 0);
        }

        float scale;
        float offset;
    };

    class BIQuantizationInfo {
    public:
        BIQuantizationInfo() noexcept: _scale(), _offset() {

        }

        BIQuantizationInfo(float scale) : _scale(1, scale), _offset() {

        }

        BIQuantizationInfo(float scale,
                           int offset,
                           bool is_dynamic = false) :
                _scale(1, scale),
                _offset(1, offset),
                _is_dynamic(is_dynamic) {

        }

        BIQuantizationInfo(std::vector<float> scale) : _scale(scale), _offset() {

        }

        BIQuantizationInfo(std::vector<float> scale,
                           std::vector<int32_t> offset,
                           bool is_dynamic = false) :
                _scale(scale),
                _offset(offset),
                _is_dynamic(is_dynamic) {

        }

        const std::vector<float> &scale() const {
            return _scale;
        }

        const std::vector<int32_t> &offset() const {
            return _offset;
        }

        bool is_dynamic() const {
            return _is_dynamic;
        }

        bool empty() const {
            return _scale.empty() && _offset.empty();
        }

        BIUniformQuantizationInfo uniform() const {
            BIUniformQuantizationInfo uqinfo;
            uqinfo.scale = _scale.empty() ? 0 : _scale[0];
            uqinfo.offset = _offset.empty() ? 0 : _offset[0];

            return uqinfo;
        }

    private:
        /**
         * @brief 包含缩放因子的向量
         */
        std::vector<float> _scale;

        /**
         * @brief 包含零偏移量的向量
         */
        std::vector<int32_t> _offset;

        /**
         * @brief 如果是真的，比例和偏移可能会改变，因此操作员需要在每次运行时进行读取。
         */
        bool _is_dynamic = false;

    };

    template<typename QUANTIZED_TYPE = uint8_t>
    struct BIQasymm8QuantizationHelper {
        static_assert(std::is_same<QUANTIZED_TYPE, uint8_t>::value || std::is_same<QUANTIZED_TYPE, int8_t>::value,
                      "quantized type should be either uint8_t or int8_t.");

        /** Quantize a value given a 8-bit asymmetric quantization scheme
         *
         * @param[in] value Value to quantize
         * @param[in] qinfo Quantization information to use for quantizing
         *
         * @return Quantized value
         */
        static inline QUANTIZED_TYPE quantize(float value, const BIUniformQuantizationInfo &qinfo) {
            BI_COMPUTE_ERROR_ON(qinfo.scale == 0);
            const int quantized = support::cpp11::lround(value / qinfo.scale) + qinfo.offset;
            return static_cast<QUANTIZED_TYPE>(BatmanInfer::misc::utility::clamp<decltype(quantized), QUANTIZED_TYPE>(
                    quantized));
        }

        static inline QUANTIZED_TYPE quantize(float value, const BIUniformRequantizationInfo &qinfo) {
            BI_COMPUTE_ERROR_ON(qinfo.scale == 0);
            const int quantized = support::cpp11::lround(value / qinfo.scale + qinfo.offset);
            return static_cast<QUANTIZED_TYPE>(BatmanInfer::misc::utility::clamp<decltype(quantized), QUANTIZED_TYPE>(
                    quantized));
        }

        /** Quantize a value given a 8-bit asymmetric quantization scheme using a specific rounding policy
         *
         * @param[in] value           Value to quantize
         * @param[in] qinfo           Quantization information to use for quantizing
         * @param[in] rounding_policy Rounding policy to use
         *
         * @return Quantized value
         */
        static inline QUANTIZED_TYPE
        quantize(float value, const BIUniformQuantizationInfo &qinfo, BIRoundingPolicy rounding_policy) {
            if (rounding_policy == BIRoundingPolicy::TO_NEAREST_UP) {
                return quantize(value, qinfo);
            }

            BI_COMPUTE_ERROR_ON(qinfo.scale == 0);
            const int quantized = BatmanInfer::round(value / qinfo.scale, rounding_policy) + qinfo.offset;
            return static_cast<QUANTIZED_TYPE>(BatmanInfer::misc::utility::clamp<decltype(quantized), QUANTIZED_TYPE>(
                    quantized));
        }

        static inline QUANTIZED_TYPE
        quantize(float value, const BIUniformRequantizationInfo &qinfo, BIRoundingPolicy rounding_policy) {
            if (rounding_policy == BIRoundingPolicy::TO_NEAREST_UP) {
                return quantize(value, qinfo);
            }

            BI_COMPUTE_ERROR_ON(qinfo.scale == 0);

            // We round after adding the offset, because the offset is also float
            const int quantized = BatmanInfer::round(value / qinfo.scale + qinfo.offset, rounding_policy);
            return static_cast<QUANTIZED_TYPE>(BatmanInfer::misc::utility::clamp<decltype(quantized), QUANTIZED_TYPE>(
                    quantized));
        }

        /** Quantize a value given a 8-bit asymmetric quantization scheme
         *
         * @param[in] value           Value to quantize
         * @param[in] qinfo           Quantization information to use for quantizing
         * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
         *
         * @return Quantized value
         */
        static inline QUANTIZED_TYPE
        quantize(float value, const BIQuantizationInfo &qinfo,
                 BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_NEAREST_UP) {
            const BIUniformQuantizationInfo uqinfo = qinfo.uniform();
            BI_COMPUTE_ERROR_ON(uqinfo.scale == 0);
            const int quantized = BatmanInfer::round(value / uqinfo.scale, rounding_policy) + uqinfo.offset;
            return static_cast<QUANTIZED_TYPE>(BatmanInfer::misc::utility::clamp<decltype(quantized), QUANTIZED_TYPE>(
                    quantized));
        }

        /** Dequantize a value given a 8-bit asymmetric quantization scheme
         *
         * @param[in] value Value to dequantize
         * @param[in] qinfo Quantization information to use for dequantizing
         *
         * @return Dequantized value
         */
        static inline float dequantize(QUANTIZED_TYPE value, const BIUniformQuantizationInfo &qinfo) {
            return (static_cast<int>(value) - qinfo.offset) * qinfo.scale;
        }

        /** Dequantize a value given a 8-bit asymmetric quantization scheme
         *
         * @param[in] value Value to dequantize
         * @param[in] qinfo Quantization information to use for dequantizing
         *
         * @return Dequantized value
         */
        static inline float dequantize(QUANTIZED_TYPE value, const BIQuantizationInfo &qinfo) {
            const BIUniformQuantizationInfo uqinfo = qinfo.uniform();
            return (static_cast<int>(value) - uqinfo.offset) * uqinfo.scale;
        }
    };

    /** Quantize a value given an unsigned 8-bit asymmetric quantization scheme
     *
     * @param[in] value           Value to quantize
     * @param[in] qinfo           Quantization information to use for quantizing
     * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
     *
     * @return Quantized value
     */
    template<typename INFO_TYPE>
    inline uint8_t quantize_qasymm8(float value,
                                    const INFO_TYPE &qinfo,
                                    BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_NEAREST_UP) {
        return BIQasymm8QuantizationHelper<uint8_t>::quantize(value, qinfo, rounding_policy);
    }


    /** Quantize a value given a signed 8-bit asymmetric quantization scheme
     *
     * @param[in] value           Value to quantize
     * @param[in] qinfo           Quantization information to use for quantizing
     * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
     *
     * @return Quantized value
     */
    template<typename INFO_TYPE>
    inline int8_t quantize_qasymm8_signed(float value,
                                          const INFO_TYPE &qinfo,
                                          BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_NEAREST_UP) {
        return BIQasymm8QuantizationHelper<int8_t>::quantize(value, qinfo, rounding_policy);
    }

    /** Quantize a value given a 8-bit symmetric quantization scheme
     *
     * @param[in] value Value to quantize
     * @param[in] qinfo Quantization information to use for quantizing
     *
     * @return Quantized value
     */
    inline int8_t quantize_qsymm8(float value, const BIQuantizationInfo &qinfo) {
        int quantized = BatmanInfer::round(value / qinfo.uniform().scale, BIRoundingPolicy::TO_NEAREST_UP);
        quantized = std::max(-128, std::min(quantized, 127));
        return quantized;
    }

    /** Quantize a value given a 16-bit asymmetric quantization scheme
     *
     * @param[in] value           Value to quantize
     * @param[in] qinfo           Quantization information to use for quantizing
     * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
     *
     * @return Quantized value
     */
    inline uint16_t quantize_qasymm16(float value,
                                      const BIUniformQuantizationInfo &qinfo,
                                      BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_NEAREST_UP) {
        int quantized = BatmanInfer::round(value / qinfo.scale, rounding_policy) + qinfo.offset;
        quantized = BatmanInfer::misc::utility::clamp<int, uint16_t>(quantized);
        return quantized;
    }

    /** Quantize a value given a 16-bit asymmetric quantization scheme
     *
     * @param[in] value Value to quantize
     * @param[in] qinfo Quantization information to use for quantizing
     *
     * @return Quantized value
     */
    inline uint16_t quantize_qasymm16(float value, const BIQuantizationInfo &qinfo) {
        return quantize_qasymm16(value, qinfo.uniform());
    }

    /** Quantize a value given a 16-bit symmetric quantization scheme
     *
     * @param[in] value           Value to quantize
     * @param[in] qinfo           Quantization information to use for quantizing
     * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
     *
     * @return Quantized value
     */
    inline int16_t quantize_qsymm16(float value,
                                    const BIUniformQuantizationInfo &qinfo,
                                    BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_NEAREST_UP) {
        int quantized = BatmanInfer::round(value / qinfo.scale, rounding_policy);
        quantized = BatmanInfer::misc::utility::clamp<int, int16_t>(quantized);
        return quantized;
    }

    /** Quantize a value given a 16-bit symmetric quantization scheme
     *
     * @param[in] value Value to quantize
     * @param[in] qinfo Quantization information to use for quantizing
     *
     * @return Quantized value
     */
    inline int16_t quantize_qsymm16(float value, const BIQuantizationInfo &qinfo) {
        return quantize_qsymm16(value, qinfo.uniform());
    }

    /** Check whether two quantization info are equal.
 *
 * @param[in] lhs RHS quantization info.
 * @param[in] rhs LHS quantization info.
 *
 * @return True if the given quantization info is the same.
 */
    inline bool operator==(const BIUniformQuantizationInfo &lhs, const BIUniformQuantizationInfo &rhs) {
        return (lhs.scale == rhs.scale) && (lhs.offset == rhs.offset);
    }

    /** Quantize a value given a 8-bit symmetric per channel quantization scheme
 *
 * @param[in] value      Value to quantize
 * @param[in] qinfo      Quantization information to use for quantizing
 * @param[in] channel_id channel index into the scale vector of quantization info
 *
 * @return Quantized value
 */
    inline int8_t quantize_qsymm8_per_channel(float value, const BIQuantizationInfo &qinfo, size_t channel_id = 0) {
        int quantized = BatmanInfer::round(value / qinfo.scale()[channel_id], BIRoundingPolicy::TO_NEAREST_UP);
        quantized = std::max(-128, std::min(quantized, 127));
        return quantized;
    }

/** Dequantize a value given an unsigned 8-bit asymmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
    template<typename INFO_TYPE>
    inline float dequantize_qasymm8(uint8_t value, const INFO_TYPE &qinfo) {
        return BIQasymm8QuantizationHelper<uint8_t>::dequantize(value, qinfo);
    }

/** Dequantize a value given a signed 8-bit asymmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
    template<typename INFO_TYPE>
    inline float dequantize_qasymm8_signed(int8_t value, const INFO_TYPE &qinfo) {
        return BIQasymm8QuantizationHelper<int8_t>::dequantize(value, qinfo);
    }

    inline float dequantize_qsymm16(int16_t value, const BIUniformQuantizationInfo &qinfo) {
        return value * qinfo.scale;
    }

/** Dequantize a value given a 16-bit symmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
    inline float dequantize_qsymm16(int16_t value, const BIQuantizationInfo &qinfo) {
        return dequantize_qsymm16(value, qinfo.uniform());
    }

    inline bool operator==(const BIQuantizationInfo &lhs, const BIQuantizationInfo &rhs) {
        return (lhs.scale() == rhs.scale()) && (lhs.offset() == rhs.offset());
    }

/** Check whether two quantization info are not equal.
 *
 * @param[in] lhs RHS quantization info.
 * @param[in] rhs LHS quantization info.
 *
 * @return True if the given quantization info is the same.
 */
    inline bool operator!=(const BIQuantizationInfo &lhs, const BIQuantizationInfo &rhs) {
        return !(operator==(lhs, rhs));
    }

    /** Compute the requantization offset and scale
 *
 * @deprecated because reequantization using integer offsets creates rounding issues.
 * Please use @ref arm_compute::compute_requantization_scale_float_offset() instead.
 *
 * In case of requantization of a quantized input tensor to an output tensor with another quantization
 * instead of applying dequantization and then a quantization functions, we just compute new scale and
 * offset.
 *
 * Assuming:
 *   - q_i as input quantized value
 *   - q_o as output quantized value
 *   - z_i as input quantization offset value
 *   - z_o as output quantization offset value
 *   - s_i as input quantization scale value
 *   - s_o as output quantization scale value
 *   - z_n as new quantization offset value
 *   - s_n as new quantization scale value
 *
 * q_o = ( q_i - z_i ) * s_i / s_o + z_o
 *
 * We can rewrite the formula as:
 *
 * q_o = ( q_i * s_i / s_o ) - z_i * s_i / s_o + z_o
 *
 * q_o = q_i / s_n + z_n
 *
 * Where:
 *
 * s_n = s_o / s_i
 *
 * z_n = - z_i * s_i / s_o + z_o
 *
 */
    inline BIUniformQuantizationInfo compute_requantization_scale_offset(const BIUniformQuantizationInfo &uqinfo_in,
                                                                         const BIUniformQuantizationInfo &uqinfo_out) {
        float scale_to_apply = uqinfo_out.scale;
        int32_t offset_to_apply = uqinfo_out.offset;

        scale_to_apply /= uqinfo_in.scale;
        // In order to minimize flooring we convert the offset to a float,
        // then compute the new offset in the float domain,
        // finally we convert it back as int32_t

#ifdef __aarch64__
        constexpr BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
        constexpr BIRoundingPolicy rounding_policy = BIRoundingPolicy::TO_NEAREST_UP;
#endif //__aarch64__

        offset_to_apply -=
                BatmanInfer::round(static_cast<float>(uqinfo_in.offset) * uqinfo_in.scale / uqinfo_out.scale,
                                   rounding_policy);
        return BIUniformQuantizationInfo(scale_to_apply, offset_to_apply);
    }

    /** Similar to @ref arm_compute::compute_requantization_scale_offset()
     *  but returning offset as float instead of integer
    */
    inline BIUniformRequantizationInfo
    compute_requantization_scale_float_offset(const BIUniformQuantizationInfo &uqinfo_in,
                                              const BIUniformQuantizationInfo &uqinfo_out) {
        float scale_to_apply = uqinfo_out.scale;
        float offset_to_apply = static_cast<float>(uqinfo_out.offset);

        scale_to_apply /= uqinfo_in.scale;
        offset_to_apply -= static_cast<float>(uqinfo_in.offset) * uqinfo_in.scale / uqinfo_out.scale;

        return BIUniformRequantizationInfo(scale_to_apply, offset_to_apply);
    }

    /**
     * Dequantize a value given a 8-bit symmetric quantization scheme
     * @param value
     * @param scale
     * @return
     */
    inline float dequantize(int8_t value, float scale) {
        return value * scale;
    }
}

#endif //BATMANINFER_QUANTIZATION_INFO_HPP
