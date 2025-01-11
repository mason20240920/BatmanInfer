//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_DATA_TYPE_UTILS_HPP
#define BATMANINFER_DATA_TYPE_UTILS_HPP

#include <data/core/bi_types.hpp>
#include <map>
#include <data/core/bi_pixel_value.h>

namespace BatmanInfer {

    const std::string &string_from_data_type(BIDataType dt);

    /**
     * 根据输入的数据类型（BIDataType）返回该数据类型的 大小（以字节为单位）
     * @param data_type 枚举类型 BIDataType，表示不同的数据类型
     * @return
     */
    inline size_t data_size_from_type(BIDataType data_type) {
        switch (data_type) {
            // 1 字节（8 位）数据类型
            case BIDataType::U8:
            case BIDataType::S8:
            case BIDataType::QSYMM8:
            case BIDataType::QASYMM8:
            case BIDataType::QASYMM8_SIGNED:
            case BIDataType::QSYMM8_PER_CHANNEL:
                return 1;
                // 2 字节（16 位）数据类型
            case BIDataType::U16:
            case BIDataType::S16:
            case BIDataType::QSYMM16:
            case BIDataType::QASYMM16:
            case BIDataType::BFLOAT16:
            case BIDataType::F16:
                return 2;
                // 4 字节（32 位）数据类型:
            case BIDataType::F32:
            case BIDataType::U32:
            case BIDataType::S32:
                return 4;
                // 8 字节（64 位）数据类型
            case BIDataType::F64:
            case BIDataType::U64:
            case BIDataType::S64:
                return 8;
                // 平台相关的类型
            case BIDataType::SIZET:
                return sizeof(size_t);
            default:
                BI_COMPUTE_ERROR("Invalid data type");
                return 0;
        }
    }

    inline BIDataType data_type_from_format(Format format) {
        switch (format) {
            case Format::U8:
            case Format::UV88:
            case Format::RGB888:
            case Format::RGBA8888:
            case Format::YUYV422:
            case Format::UYVY422:
                return BIDataType::U8;
            case Format::U16:
                return BIDataType::U16;
            case Format::S16:
                return BIDataType::S16;
            case Format::U32:
                return BIDataType::U32;
            case Format::S32:
                return BIDataType::S32;
            case Format::BFLOAT16:
                return BIDataType::BFLOAT16;
            case Format::F16:
                return BIDataType::F16;
            case Format::F32:
                return BIDataType::F32;
                //Doesn't make sense for planar formats:
            case Format::NV12:
            case Format::NV21:
            case Format::IYUV:
            case Format::YUV444:
            default:
                BI_COMPUTE_ERROR("Not supported data_type for given format");
                return BIDataType::UNKNOWN;
        }
    }

    /**
     * @brief  检查给定的数据类型是否为浮点类型。
     * @param dt
     * @return
     */
    inline bool is_data_type_float(BIDataType dt) {
        switch (dt) {
            case BIDataType::F16:
            case BIDataType::F32:
                return true;
            default:
                return false;
        }
    }

    /**
     * 检查数据类型是不是非对称量化
     * @param dt
     * @return
     */
    inline bool is_data_type_quantized_asymmetric(BIDataType dt) {
        switch (dt) {
            case BIDataType::QASYMM8:
            case BIDataType::QASYMM8_SIGNED:
            case BIDataType::QASYMM16:
                return true;
            default:
                return false;
        }
    }

    inline bool is_data_type_quantized_asymmetric_char(BIDataType dt) {
        switch (dt) {
            case BIDataType::QASYMM8_SIGNED:
            case BIDataType::QASYMM8:
                return true;
            default:
                return false;
        }
    }


    /**
     * 检查是否对称量化
     * @param dt
     * @return
     */
    inline bool is_data_type_quantized_symmetric(BIDataType dt) {
        switch (dt) {
            case BIDataType::QSYMM8:
            case BIDataType::QSYMM8_PER_CHANNEL:
            case BIDataType::QSYMM16:
                return true;
            default:
                return false;
        }
    }


    inline bool is_data_type_quantized(BIDataType dt) {
        switch (dt) {
            case BIDataType::QSYMM8:
            case BIDataType::QASYMM8:
            case BIDataType::QASYMM8_SIGNED:
            case BIDataType::QSYMM8_PER_CHANNEL:
            case BIDataType::QSYMM16:
            case BIDataType::QASYMM16:
                return true;
            default:
                return false;
        }
    }

    inline std::tuple<BIPixelValue, BIPixelValue> get_min_max(BIDataType dt) {
        BIPixelValue min{};
        BIPixelValue max{};
        switch (dt) {
            case BIDataType::U8:
            case BIDataType::QASYMM8: {
                min = BIPixelValue(static_cast<int32_t>(std::numeric_limits<uint8_t>::lowest()));
                max = BIPixelValue(static_cast<int32_t>(std::numeric_limits<uint8_t>::max()));
                break;
            }
            case BIDataType::S8:
            case BIDataType::QSYMM8:
            case BIDataType::QASYMM8_SIGNED:
            case BIDataType::QSYMM8_PER_CHANNEL: {
                min = BIPixelValue(static_cast<int32_t>(std::numeric_limits<int8_t>::lowest()));
                max = BIPixelValue(static_cast<int32_t>(std::numeric_limits<int8_t>::max()));
                break;
            }
            case BIDataType::U16:
            case BIDataType::QASYMM16: {
                min = BIPixelValue(static_cast<int32_t>(std::numeric_limits<uint16_t>::lowest()));
                max = BIPixelValue(static_cast<int32_t>(std::numeric_limits<uint16_t>::max()));
                break;
            }
            case BIDataType::S16:
            case BIDataType::QSYMM16: {
                min = BIPixelValue(static_cast<int32_t>(std::numeric_limits<int16_t>::lowest()));
                max = BIPixelValue(static_cast<int32_t>(std::numeric_limits<int16_t>::max()));
                break;
            }
            case BIDataType::U32: {
                min = BIPixelValue(std::numeric_limits<uint32_t>::lowest());
                max = BIPixelValue(std::numeric_limits<uint32_t>::max());
                break;
            }
            case BIDataType::S32: {
                min = BIPixelValue(std::numeric_limits<int32_t>::lowest());
                max = BIPixelValue(std::numeric_limits<int32_t>::max());
                break;
            }
            case BIDataType::BFLOAT16: {
                min = BIPixelValue(bfloat16::lowest());
                max = BIPixelValue(bfloat16::max());
                break;
            }
            case BIDataType::F16: {
                min = BIPixelValue(std::numeric_limits<half>::lowest());
                max = BIPixelValue(std::numeric_limits<half>::max());
                break;
            }
            case BIDataType::F32: {
                min = BIPixelValue(std::numeric_limits<float>::lowest());
                max = BIPixelValue(std::numeric_limits<float>::max());
                break;
            }
            default:
                BI_COMPUTE_ERROR("Undefined data type!");
        }
        return std::make_tuple(min, max);
    }
}

#endif //BATMANINFER_DATA_TYPE_UTILS_HPP
