//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_DATA_TYPE_UTILS_HPP
#define BATMANINFER_DATA_TYPE_UTILS_HPP

#include <data/core/bi_types.hpp>

namespace BatmanInfer {
    inline size_t data_size_from_type(BIDataType data_type)
    {
        switch (data_type)
        {
            case BIDataType::U8:
            case BIDataType::S8:
            case BIDataType::QSYMM8:
            case BIDataType::QASYMM8:
            case BIDataType::QASYMM8_SIGNED:
            case BIDataType::QSYMM8_PER_CHANNEL:
                return 1;
            case BIDataType::U16:
            case BIDataType::S16:
            case BIDataType::QSYMM16:
            case BIDataType::QASYMM16:
            case BIDataType::BFLOAT16:
            case BIDataType::F16:
                return 2;
            case BIDataType::F32:
            case BIDataType::U32:
            case BIDataType::S32:
                return 4;
            case BIDataType::F64:
            case BIDataType::U64:
            case BIDataType::S64:
                return 8;
            case BIDataType::SIZET:
                return sizeof(size_t);
            default:
                ARM_COMPUTE_ERROR("Invalid data type");
                return 0;
        }
    }

    inline BIDataType data_type_from_format(Format format)
    {
        switch (format)
        {
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
                ARM_COMPUTE_ERROR("Not supported data_type for given format");
                return BIDataType::UNKNOWN;
        }
    }
}

#endif //BATMANINFER_DATA_TYPE_UTILS_HPP
