//
// Created by Mason on 2024/12/25.
//

#ifndef BATMANINFER_CORE_TYPES_HPP
#define BATMANINFER_CORE_TYPES_HPP

namespace BatmanInfer {

    using TensorID   = unsigned int;
    /**
 * @brief 可用的数据类型
 */
    enum class BIDataType {
        UNKNOWN,            /**< Unknown data type */
        U8,                 /**< unsigned 8-bit number */
        S8,                 /**< signed 8-bit number */
        QSYMM8,             /**< quantized, symmetric fixed-point 8-bit number */
        QASYMM8,            /**< quantized, asymmetric fixed-point 8-bit number unsigned */
        QASYMM8_SIGNED,     /**< quantized, asymmetric fixed-point 8-bit number signed */
        QSYMM8_PER_CHANNEL, /**< quantized, symmetric per channel fixed-point 8-bit number */
        U16,                /**< unsigned 16-bit number */
        S16,                /**< signed 16-bit number */
        QSYMM16,            /**< quantized, symmetric fixed-point 16-bit number */
        QASYMM16,           /**< quantized, asymmetric fixed-point 16-bit number */
        U32,                /**< unsigned 32-bit number */
        S32,                /**< signed 32-bit number */
        U64,                /**< unsigned 64-bit number */
        S64,                /**< signed 64-bit number */
        BFLOAT16,           /**< 16-bit brain floating-point number */
        F16,                /**< 16-bit floating-point number */
        F32,                /**< 32-bit floating-point number */
        F64,                /**< 64-bit floating-point number */
        SIZET               /**< size_t */
    };

    /** Image colour formats */
    enum class Format
    {
        UNKNOWN,  /**< Unknown image format */
        U8,       /**< 1 channel, 1 U8 per channel */
        S16,      /**< 1 channel, 1 S16 per channel */
        U16,      /**< 1 channel, 1 U16 per channel */
        S32,      /**< 1 channel, 1 S32 per channel */
        U32,      /**< 1 channel, 1 U32 per channel */
        S64,      /**< 1 channel, 1 S64 per channel */
        U64,      /**< 1 channel, 1 U64 per channel */
        BFLOAT16, /**< 16-bit brain floating-point number */
        F16,      /**< 1 channel, 1 F16 per channel */
        F32,      /**< 1 channel, 1 F32 per channel */
        UV88,     /**< 2 channel, 1 U8 per channel */
        RGB888,   /**< 3 channels, 1 U8 per channel */
        RGBA8888, /**< 4 channels, 1 U8 per channel */
        YUV444,   /**< A 3 plane of 8 bit 4:4:4 sampled Y, U, V planes */
        YUYV422,  /**< A single plane of 32-bit macro pixel of Y0, U0, Y1, V0 bytes */
        NV12,     /**< A 2 plane YUV format of Luma (Y) and interleaved UV data at 4:2:0 sampling */
        NV21,     /**< A 2 plane YUV format of Luma (Y) and interleaved VU data at 4:2:0 sampling */
        IYUV,     /**< A 3 plane of 8-bit 4:2:0 sampled Y, U, V planes */
        UYVY422   /**< A single plane of 32-bit macro pixel of U0, Y0, V0, Y1 byte */
    };

/**< Device target types */
    enum class BITarget
    {
        UNSPECIFIED, /**< Unspecified Target */
        NEON,        /**< Arm® Neon™ capable target device */
        CL,          /**< OpenCL capable target device */
        CUDA
    };

    /** Supported tensor data layout dimensions */
    enum class BIDataLayoutDimension
    {
        CHANNEL, /**< channel */
        HEIGHT,  /**< height */
        WIDTH,   /**< width */
        DEPTH,   /**< depth */
        BATCHES  /**< batches */
    };
}

#endif //BATMANINFER_CORE_TYPES_HPP
