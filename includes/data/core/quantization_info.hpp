//
// Created by Mason on 2024/12/25.
//

#ifndef BATMANINFER_QUANTIZATION_INFO_HPP
#define BATMANINFER_QUANTIZATION_INFO_HPP

#include <vector>

namespace BatmanInfer {

    using qasymm8_signed_t = int8_t;   /**< 8 bit signed quantized asymmetric scalar value */
    using qasymm8_t        = uint8_t;  /**< 8 bit quantized asymmetric scalar value */
    using qsymm16_t        = int16_t;  /**< 16 bit quantized symmetric scalar value */
    using qasymm16_t       = uint16_t; /**< 16 bit quantized asymmetric scalar value */

    /** Quantization info when assuming per layer quantization */
    struct BIUniformQuantizationInfo
    {
        /** Default constructor */
        BIUniformQuantizationInfo() : scale(0.f), offset(0)
        {
        }
        /** Constructor
         *
         * @param[in] scale  Quantization scale
         * @param[in] offset Quantization offset
         */
        BIUniformQuantizationInfo(float scale, int32_t offset) : scale(scale), offset(offset)
        {
        }
        /** Checks if the scale and offset are both zero */
        bool empty() const
        {
            return (scale == 0) && (offset == 0);
        }

        float   scale;
        int32_t offset;
    };


    class BIQuantizationInfo {
    public:
        BIQuantizationInfo() noexcept: _scale(), _offset()
        {

        }

        BIQuantizationInfo(float scale) : _scale(1, scale), _offset()
        {

        }

        BIQuantizationInfo(float scale,
                           int offset,
                           bool is_dynamic = false):
                _scale(1, scale),
                _offset(1, offset),
                _is_dynamic(is_dynamic)
        {

        }

        BIQuantizationInfo(std::vector<float> scale): _scale(scale), _offset()
        {

        }

        BIQuantizationInfo(std::vector<float> scale,
                           std::vector<int32_t> offset,
                           bool is_dynamic = false) :
                _scale(scale),
                _offset(offset),
                _is_dynamic(is_dynamic)
        {

        }

        const std::vector<float> &scale() const
        {
            return _scale;
        }

        const std::vector<int32_t> &offset() const
        {
            return _offset;
        }

        bool is_dynamic() const
        {
            return _is_dynamic;
        }

        bool empty() const
        {
            return _scale.empty() && _offset.empty();
        }

        BIUniformQuantizationInfo uniform() const {
            BIUniformQuantizationInfo uqinfo;
            uqinfo.scale  = _scale.empty() ? 0 : _scale[0];
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
}

#endif //BATMANINFER_QUANTIZATION_INFO_HPP
