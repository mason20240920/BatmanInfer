//
// Created by Mason on 2024/12/25.
//

#ifndef BATMANINFER_CORE_TYPES_HPP
#define BATMANINFER_CORE_TYPES_HPP

#include "data/core/bi_strides.hpp"

#include "support/bi_half.hpp"

namespace BatmanInfer {
    /** 16-bit floating point type */
    using half = half_float::half;

    using TensorID = unsigned int;
    /** Permutation vector */
    using PermutationVector = BIStrides;

    /**
 * @brief 可用的数据类型
 */
    enum class BIDataType {
        UNKNOWN, /**< Unknown data type */
        U8, /**< unsigned 8-bit number */
        S8, /**< signed 8-bit number */
        QSYMM8, /**< quantized, symmetric fixed-point 8-bit number */
        QASYMM8, /**< quantized, asymmetric fixed-point 8-bit number unsigned */
        QASYMM8_SIGNED, /**< quantized, asymmetric fixed-point 8-bit number signed */
        QSYMM8_PER_CHANNEL, /**< quantized, symmetric per channel fixed-point 8-bit number */
        U16, /**< unsigned 16-bit number */
        S16, /**< signed 16-bit number */
        QSYMM16, /**< quantized, symmetric fixed-point 16-bit number */
        QASYMM16, /**< quantized, asymmetric fixed-point 16-bit number */
        U32, /**< unsigned 32-bit number */
        S32, /**< signed 32-bit number */
        U64, /**< unsigned 64-bit number */
        S64, /**< signed 64-bit number */
        BFLOAT16, /**< 16-bit brain floating-point number */
        F16, /**< 16-bit floating-point number */
        F32, /**< 32-bit floating-point number */
        F64, /**< 64-bit floating-point number */
        SIZET /**< size_t */
    };

    /** Supported tensor data layouts */
    enum class BIDataLayout {
        UNKNOWN, /**< Unknown data layout */
        NCHW, /**< Num samples, channels, height, width */
        NHWC, /**< Num samples, height, width, channels */
        NCDHW, /**< Num samples, channels, depth, height, width */
        NDHWC /**< Num samples, depth, height, width, channels */
    };

    /** Image colour formats */
    enum class Format {
        UNKNOWN, /**< Unknown image format */
        U8, /**< 1 channel, 1 U8 per channel */
        S16, /**< 1 channel, 1 S16 per channel */
        U16, /**< 1 channel, 1 U16 per channel */
        S32, /**< 1 channel, 1 S32 per channel */
        U32, /**< 1 channel, 1 U32 per channel */
        S64, /**< 1 channel, 1 S64 per channel */
        U64, /**< 1 channel, 1 U64 per channel */
        BFLOAT16, /**< 16-bit brain floating-point number */
        F16, /**< 1 channel, 1 F16 per channel */
        F32, /**< 1 channel, 1 F32 per channel */
        UV88, /**< 2 channel, 1 U8 per channel */
        RGB888, /**< 3 channels, 1 U8 per channel */
        RGBA8888, /**< 4 channels, 1 U8 per channel */
        YUV444, /**< A 3 plane of 8 bit 4:4:4 sampled Y, U, V planes */
        YUYV422, /**< A single plane of 32-bit macro pixel of Y0, U0, Y1, V0 bytes */
        NV12, /**< A 2 plane YUV format of Luma (Y) and interleaved UV data at 4:2:0 sampling */
        NV21, /**< A 2 plane YUV format of Luma (Y) and interleaved VU data at 4:2:0 sampling */
        IYUV, /**< A 3 plane of 8-bit 4:2:0 sampled Y, U, V planes */
        UYVY422, /**< A single plane of 32-bit macro pixel of U0, Y0, V0, Y1 byte */
        S8
    };

    /**< Device target types */
    enum class BITarget {
        UNSPECIFIED, /**< Unspecified Target */
        NEON, /**< Arm® Neon™ capable target device */
        CL, /**< OpenCL capable target device */
        CUDA
    };

    /** Supported tensor data layout dimensions */
    enum class BIDataLayoutDimension {
        CHANNEL, /**< channel */
        HEIGHT, /**< height */
        WIDTH, /**< width */
        DEPTH, /**< depth */
        BATCHES /**< batches */
    };

    /** Dimension rounding type when down-scaling on CNNs
     * @note Used in pooling and convolution layer
     */
    enum class BIDimensionRoundingType {
        FLOOR, /**< Floor rounding */
        CEIL /**< Ceil rounding */
    };

    class BIPadStrideInfo {
    public:
        /** Constructor
         *
         * @param[in] stride_x (Optional) Stride, in elements, across x. Defaults to 1.
         * @param[in] stride_y (Optional) Stride, in elements, across y. Defaults to 1.
         * @param[in] pad_x    (Optional) Padding, in elements, across x. Defaults to 0.
         * @param[in] pad_y    (Optional) Padding, in elements, across y. Defaults to 0.
         * @param[in] round    (Optional) Dimensions rounding. Defaults to @ref BIDimensionRoundingType::FLOOR.
         */
        BIPadStrideInfo(unsigned int stride_x = 1,
                        unsigned int stride_y = 1,
                        unsigned int pad_x = 0,
                        unsigned int pad_y = 0,
                        BIDimensionRoundingType round = BIDimensionRoundingType::FLOOR)
            : _stride(std::make_pair(stride_x, stride_y)),
              _pad_left(pad_x),
              _pad_top(pad_y),
              _pad_right(pad_x),
              _pad_bottom(pad_y),
              _round_type(round) {
        }

        /** Constructor
         *
         * @param[in] stride_x   Stride, in elements, across x.
         * @param[in] stride_y   Stride, in elements, across y.
         * @param[in] pad_left   Padding across x on the left, in elements.
         * @param[in] pad_right  Padding across x on the right, in elements.
         * @param[in] pad_top    Padding across y on the top, in elements.
         * @param[in] pad_bottom Padding across y on the bottom, in elements.
         * @param[in] round      Dimensions rounding.
         */
        BIPadStrideInfo(unsigned int stride_x,
                        unsigned int stride_y,
                        unsigned int pad_left,
                        unsigned int pad_right,
                        unsigned int pad_top,
                        unsigned int pad_bottom,
                        BIDimensionRoundingType round)
            : _stride(std::make_pair(stride_x, stride_y)),
              _pad_left(pad_left),
              _pad_top(pad_top),
              _pad_right(pad_right),
              _pad_bottom(pad_bottom),
              _round_type(round) {
        }

        /** Get the stride.
         *
         * @return a pair: stride x, stride y.
         */
        std::pair<unsigned int, unsigned int> stride() const {
            return _stride;
        }

        /** Check whether the padding is symmetric.
         *
         * @return True if the padding is symmetric.
         */
        bool padding_is_symmetric() const {
            return (_pad_left == _pad_right) && (_pad_top == _pad_bottom);
        }

        /** Get the padding.
         *
         * @note This should only be used when the padding is symmetric.
         *
         * @return a pair: padding left/right, padding top/bottom
         */
        std::pair<unsigned int, unsigned int> pad() const {
            //this accessor should be used only when padding is symmetric
            BI_COMPUTE_ERROR_ON(!padding_is_symmetric());
            return std::make_pair(_pad_left, _pad_top);
        }

        /** Get the left padding */
        unsigned int pad_left() const {
            return _pad_left;
        }

        /** Get the right padding */
        unsigned int pad_right() const {
            return _pad_right;
        }

        /** Get the top padding */
        unsigned int pad_top() const {
            return _pad_top;
        }

        /** Get the bottom padding */
        unsigned int pad_bottom() const {
            return _pad_bottom;
        }

        /** Get the rounding type */
        BIDimensionRoundingType round() const {
            return _round_type;
        }

        /** Check whether this has any padding */
        bool has_padding() const {
            return (_pad_left != 0 || _pad_top != 0 || _pad_right != 0 || _pad_bottom != 0);
        }

    private:
        std::pair<unsigned int, unsigned int> _stride;
        unsigned int _pad_left;
        unsigned int _pad_top;
        unsigned int _pad_right;
        unsigned int _pad_bottom;

        BIDimensionRoundingType _round_type;
    };

    /** 权重张量的内存布局
    *
    * * UNSPECIFIED（未指定）：用于选择不支持“可变权重模式”（variable weights mode）的内核。
    *
    * * ANY：用于查询内核数据库，获取任何支持“可变权重模式”的内核。
    *   一旦找到合适的内核，用户可以获取该内核所需的特定权重张量格式，并相应地重新排列权重张量。
    *
    * 其他值（如 OHWIo{interleave_by}i{block_by}）描述了一个 4D 张量（布局为 OHWI）的内存布局，
    * 它被转换为另一个 4D 张量（尺寸为 O'HWI'），其中：
    *
    * O' = 满足 O <= O' 的第一个 {interleave_by} 的倍数
    * I' = 满足 I <= I' 的第一个 {block_by} 的倍数
    *
    * 转换后的目标张量（dst tensor）的总大小为 O' x H x W x I'。
    *
    * 对于布局为 OHWIo{interleave_by}i{block_by} 且尺寸为 O'HWI' 的张量，其访问函数为一个 6 参数的访问函数，
    * 6 个参数的计算方式如下：
    *
    * x5 = floor(io/{interleave_by})  范围：[0, O'/{interleave_by} - 1]  大小：O'/{interleave_by}
    *
    * x4 = h                         范围：[0, H - 1]                   大小：H
    * x3 = w                         范围：[0, W - 1]                   大小：W
    * x2 = floor(i/{block_by})       范围：[0, I'/{block_by} - 1]       大小：I'/{block_by}
    * x1 = io%{interleave_by}        范围：[0, {interleave_by} - 1]     大小：{interleave_by}
    * x0 = i%{block_by}              范围：[0, {block_by} - 1]          大小：{block_by}
    *                                                              总大小：O' * H * W * I'
    *
    *        4D 张量                  6D 张量
    * -----------------   -----------------------------------
    * value(io, h, w, i) =   x5 * H * W * I' * {interleave_by}
    *                     + x4 * W * I' * {interleave_by}
    *                     + x3 * I' * {interleave_by}
    *                     + x2 * {interleave_by} * {block_by}
    *                     + x1 * {block_by}
    *                     + x0
    *
    * 注意，在 BatmanGemm 中，尺寸为 O'HWI' 的 4D 张量（布局为 OHWIo{interleave_by}i{block_by}）实际上被视为一个 2D 张量，
    * 其中行数为 O'/{interleave_by}，列数为 {interleave_by} * H * W * I'。
    *
    * 后缀 *_bf16 表示该内存布局是为快速模式（fast-mode）内核所需的，
    * 在这种模式下，权重以 bfloat16 格式传递。
    */
    enum class BIWeightFormat {
        UNSPECIFIED = 0x1,
        ANY = 0x2,
        OHWI = 0x100100,
        OHWIo2 = 0x100200,
        OHWIo4 = 0x100400,
        OHWIo8 = 0x100800,
        OHWIo16 = 0x101000,
        OHWIo32 = 0x102000,
        OHWIo64 = 0x104000,
        OHWIo128 = 0x108000,
        OHWIo4i2 = 0x200400,
        OHWIo4i2_bf16 = 0x200410,
        OHWIo8i2 = 0x200800,
        OHWIo8i2_bf16 = 0x200810,
        OHWIo16i2 = 0x201000,
        OHWIo16i2_bf16 = 0x201010,
        OHWIo32i2 = 0x202000,
        OHWIo32i2_bf16 = 0x202010,
        OHWIo64i2 = 0x204000,
        OHWIo64i2_bf16 = 0x204010,
        OHWIo4i4 = 0x400400,
        OHWIo4i4_bf16 = 0x400410,
        OHWIo8i4 = 0x400800,
        OHWIo8i4_bf16 = 0x400810,
        OHWIo16i4 = 0x401000,
        OHWIo16i4_bf16 = 0x401010,
        OHWIo32i4 = 0x402000,
        OHWIo32i4_bf16 = 0x402010,
        OHWIo64i4 = 0x404000,
        OHWIo64i4_bf16 = 0x404010,
        OHWIo2i8 = 0x800200,
        OHWIo4i8 = 0x800400,
        OHWIo8i8 = 0x800800,
        OHWIo16i8 = 0x801000,
        OHWIo32i8 = 0x802000,
        OHWIo64i8 = 0x804000
    };
} // namespace BatmanInfer

#endif //BATMANINFER_CORE_TYPES_HPP
