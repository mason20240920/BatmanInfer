//
// Created by holynova on 2025/1/3.
//

#ifndef BATMANINFER_BI_PIXEL_VALUE_H
#define BATMANINFER_BI_PIXEL_VALUE_H

#include "data/core/quantization_info.hpp"
#include "data/core/bi_types.hpp"

#include <cstdint>

namespace BatmanInfer {

    /** Class describing the value of a pixel for any image format. */
    class BIPixelValue
    {
    public:
        /** Default constructor: value initialized to 0 */
        BIPixelValue() noexcept : value{int64_t(0)}
        {
        }
        /** Initialize the union with a pixel value of chosen datatype
         *
         * @param[in] v        value.
         * @param[in] datatype DataType that @p v have to be stored
         * @param[in] qinfo    (Optional) QuantizationInfo to apply in case of quantized data types to @p v
         */
        BIPixelValue(double v, BIDataType datatype, BIQuantizationInfo qinfo = BIQuantizationInfo()) : BIPixelValue()
        {
            switch (datatype)
            {
                case BIDataType::U8:
                    value.u8 = static_cast<uint8_t>(v);
                    break;
                case BIDataType::S8:
                    value.s8 = static_cast<int8_t>(v);
                    break;
                case BIDataType::QASYMM8:
                    value.u8 = quantize_qasymm8(static_cast<float>(v), qinfo);
                    break;
                case BIDataType::QASYMM8_SIGNED:
                    value.s8 = quantize_qasymm8_signed(static_cast<float>(v), qinfo);
                    break;
                case BIDataType::QSYMM8:
                    value.s8 = quantize_qsymm8(static_cast<float>(v), qinfo);
                    break;
                case BIDataType::U16:
                    value.u16 = static_cast<uint16_t>(v);
                    break;
                case BIDataType::S16:
                    value.s16 = static_cast<int16_t>(v);
                    break;
                case BIDataType::QASYMM16:
                    value.u16 = quantize_qasymm16(static_cast<float>(v), qinfo);
                    break;
                case BIDataType::QSYMM16:
                    value.s16 = quantize_qsymm16(static_cast<float>(v), qinfo);
                    break;
                case BIDataType::U32:
                    value.u32 = static_cast<uint32_t>(v);
                    break;
                case BIDataType::S32:
                    value.s32 = static_cast<int32_t>(v);
                    break;
                case BIDataType::U64:
                    value.u64 = static_cast<uint64_t>(v);
                    break;
                case BIDataType::S64:
                    value.s64 = static_cast<int64_t>(v);
                    break;
                case BIDataType::BFLOAT16:
                    value.bf16 = static_cast<bfloat16>(v);
                    break;
                case BIDataType::F16:
                    value.f16 = static_cast<half>(v);
                    break;
                case BIDataType::F32:
                    value.f32 = static_cast<float>(v);
                    break;
                case BIDataType::F64:
                default:
                    value.f64 = v;
                    break;
            }
        }
        /** Initialize the union with a S8 pixel value
         *
         * @param[in] v S8 value.
         */
        BIPixelValue(int8_t v) : BIPixelValue()
        {
            value.s8 = v;
        }
        /** Initialize the union with a U8 pixel value
         *
         * @param[in] v U8 value.
         */
        BIPixelValue(uint8_t v) : BIPixelValue()
        {
            value.u8 = v;
        }
        /** Initialize the union with a U16 pixel value
         *
         * @param[in] v U16 value.
         */
        BIPixelValue(uint16_t v) : BIPixelValue()
        {
            value.u16 = v;
        }
        /** Initialize the union with a S16 pixel value
         *
         * @param[in] v S16 value.
         */
        BIPixelValue(int16_t v) : BIPixelValue()
        {
            value.s16 = v;
        }
        /** Initialize the union with a U32 pixel value
         *
         * @param[in] v U32 value.
         */
        BIPixelValue(uint32_t v) : BIPixelValue()
        {
            value.u32 = v;
        }
        /** Initialize the union with a S32 pixel value
         *
         * @param[in] v S32 value.
         */
        BIPixelValue(int32_t v) : BIPixelValue()
        {
            value.s32 = v;
        }

        /** Initialize the union with a U64 pixel value
         *
         * @param[in] v U64 value.
         */
        BIPixelValue(uint64_t v) : BIPixelValue()
        {
            value.u64 = v;
        }
        /** Initialize the union with a S64 pixel value
         *
         * @param[in] v S64 value.
         */
        BIPixelValue(int64_t v) : BIPixelValue()
        {
            value.s64 = v;
        }
        /** Initialize the union with a BFLOAT16 pixel value
         *
         * @param[in] v F16 value.
         */
        BIPixelValue(bfloat16 v) : BIPixelValue()
        {
            value.bf16 = v;
        }
        /** Initialize the union with a F16 pixel value
         *
         * @param[in] v F16 value.
         */
        BIPixelValue(half v) : BIPixelValue()
        {
            value.f16 = v;
        }
        /** Initialize the union with a F32 pixel value
         *
         * @param[in] v F32 value.
         */
        BIPixelValue(float v) : BIPixelValue()
        {
            value.f32 = v;
        }
        /** Initialize the union with a F64 pixel value
         *
         * @param[in] v F64 value.
         */
        BIPixelValue(double v) : BIPixelValue()
        {
            value.f64 = v;
        }
        /** Union which describes the value of a pixel for any image format.
         * Use the field corresponding to the image format
         */
        union
        {
            uint64_t u64;     /**< Single channel U64 */
            int64_t  s64;     /**< Single channel S64 */
            uint8_t  rgb[3];  /**< 3 channels: RGB888 */
            uint8_t  yuv[3];  /**< 3 channels: Any YUV format */
            uint8_t  rgbx[4]; /**< 4 channels: RGBX8888 */
            double   f64;     /**< Single channel double */
            float    f32;     /**< Single channel float 32 */
            half     f16;     /**< Single channel F16 */
            bfloat16 bf16;    /**< Single channel brain floating-point number */
            uint8_t  u8;      /**< Single channel U8 */
            int8_t   s8;      /**< Single channel S8 */
            uint16_t u16;     /**< Single channel U16 */
            int16_t  s16;     /**< Single channel S16 */
            uint32_t u32;     /**< Single channel U32 */
            int32_t  s32;     /**< Single channel S32 */
        } value;
        /** Interpret the pixel value as a U8
         *
         * @param[out] v Returned value
         */
        void get(uint8_t &v) const
        {
            v = value.u8;
        }
        /** Interpret the pixel value as a S8
         *
         * @param[out] v Returned value
         */
        void get(int8_t &v) const
        {
            v = value.s8;
        }
        /** Interpret the pixel value as a U16
         *
         * @param[out] v Returned value
         */
        void get(uint16_t &v) const
        {
            v = value.u16;
        }
        /** Interpret the pixel value as a S16
         *
         * @param[out] v Returned value
         */
        void get(int16_t &v) const
        {
            v = value.s16;
        }
        /** Interpret the pixel value as a U32
         *
         * @param[out] v Returned value
         */
        void get(uint32_t &v) const
        {
            v = value.u32;
        }
        /** Interpret the pixel value as a S32
         *
         * @param[out] v Returned value
         */
        void get(int32_t &v) const
        {
            v = value.s32;
        }
        /** Interpret the pixel value as a U64
         *
         * @param[out] v Returned value
         */
        void get(uint64_t &v) const
        {
            v = value.u64;
        }
        /** Interpret the pixel value as a S64
         *
         * @param[out] v Returned value
         */
        void get(int64_t &v) const
        {
            v = value.s64;
        }
        /** Interpret the pixel value as a BFLOAT16
         *
         * @param[out] v Returned value
         */
        void get(bfloat16 &v) const
        {
            v = value.bf16;
        }
        /** Interpret the pixel value as a F16
         *
         * @param[out] v Returned value
         */
        void get(half &v) const
        {
            v = value.f16;
        }
        /** Interpret the pixel value as a F32
         *
         * @param[out] v Returned value
         */
        void get(float &v) const
        {
            v = value.f32;
        }
        /** Interpret the pixel value as a double
         *
         * @param[out] v Returned value
         */
        void get(double &v) const
        {
            v = value.f64;
        }
        /** Get the pixel value
         *
         * @return Pixel value
         */
        template <typename T>
        T get() const
        {
            T val;
            get(val);
            return val;
        }
    };

}

#endif //BATMANINFER_BI_PIXEL_VALUE_H
