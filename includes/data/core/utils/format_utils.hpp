//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_FORMAT_UTILS_HPP
#define BATMANINFER_FORMAT_UTILS_HPP

#include <data/core/bi_types.hpp>

namespace BatmanInfer {
    /**
     * @brief
     * @param format
     * @return
     */
    inline size_t num_channels_from_format(Format format) {
        switch (format) {
            case Format::U8:
            case Format::U16:
            case Format::S16:
            case Format::U32:
            case Format::S32:
            case Format::BFLOAT16:
            case Format::F16:
            case Format::F32:
            case Format::S8:
                return 1;
            // Because the U and V channels are subsampled
            // these formats appear like having only 2 channels:
            case Format::YUYV422:
            case Format::UYVY422:
                return 2;
            case Format::UV88:
                return 2;
            case Format::RGB888:
                return 3;
            case Format::RGBA8888:
                return 4;
            //Doesn't make sense for planar formats:
            case Format::NV12:
            case Format::NV21:
            case Format::IYUV:
            case Format::YUV444:
            default:
                return 0;
        }
    }
}

#endif //BATMANINFER_FORMAT_UTILS_HPP
