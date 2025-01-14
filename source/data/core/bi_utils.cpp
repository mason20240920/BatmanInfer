//
// Created by Mason on 2025/1/2.
//

#include <data/core/bi_utils.hpp>

namespace BatmanInfer {
#ifdef BI_COMPUTE_ASSERTS_ENABLED
    void print_consecutive_elements(std::ostream      &s,
                                    BIDataType           dt,
                                    const uint8_t     *ptr,
                                    unsigned int       n,
                                    int                stream_width,
                                    const std::string &element_delim)
    {
        switch (dt)
        {
            case BIDataType::U8:
            case BIDataType::QASYMM8:
                print_consecutive_elements_impl<uint8_t>(s, ptr, n, stream_width, element_delim);
                break;
            case BIDataType::S8:
            case BIDataType::QSYMM8:
            case BIDataType::QASYMM8_SIGNED:
            case BIDataType::QSYMM8_PER_CHANNEL:
                print_consecutive_elements_impl<int8_t>(s, reinterpret_cast<const int8_t *>(ptr), n, stream_width,
                                                        element_delim);
                break;
            case BIDataType::U16:
            case BIDataType::QASYMM16:
                print_consecutive_elements_impl<uint16_t>(s, reinterpret_cast<const uint16_t *>(ptr), n, stream_width,
                                                          element_delim);
                break;
            case BIDataType::S16:
            case BIDataType::QSYMM16:
                print_consecutive_elements_impl<int16_t>(s, reinterpret_cast<const int16_t *>(ptr), n, stream_width,
                                                         element_delim);
                break;
            case BIDataType::U32:
                print_consecutive_elements_impl<uint32_t>(s, reinterpret_cast<const uint32_t *>(ptr), n, stream_width,
                                                          element_delim);
                break;
            case BIDataType::S32:
                print_consecutive_elements_impl<int32_t>(s, reinterpret_cast<const int32_t *>(ptr), n, stream_width,
                                                         element_delim);
                break;
            case BIDataType::U64:
                print_consecutive_elements_impl<uint64_t>(s, reinterpret_cast<const uint64_t *>(ptr), n, stream_width,
                                                          element_delim);
                break;
            case BIDataType::S64:
                print_consecutive_elements_impl<int64_t>(s, reinterpret_cast<const int64_t *>(ptr), n, stream_width,
                                                         element_delim);
                break;
            case BIDataType::BFLOAT16:
                print_consecutive_elements_impl<bfloat16>(s, reinterpret_cast<const bfloat16 *>(ptr), n, stream_width,
                                                          element_delim);
                break;
            case BIDataType::F16:
                print_consecutive_elements_impl<half>(s, reinterpret_cast<const half *>(ptr), n, stream_width,
                                                      element_delim);
                break;
            case BIDataType::F32:
                print_consecutive_elements_impl<float>(s, reinterpret_cast<const float *>(ptr), n, stream_width,
                                                       element_delim);
                break;
            default:
                BI_COMPUTE_ERROR("Undefined element size for given data type");
        }
    }

        int max_consecutive_elements_display_width(std::ostream &s, BIDataType dt, const uint8_t *ptr, unsigned int n)
        {
            switch (dt)
            {
                case BIDataType::U8:
                case BIDataType::QASYMM8:
                    return max_consecutive_elements_display_width_impl<uint8_t>(s, ptr, n);
                case BIDataType::S8:
                case BIDataType::QSYMM8:
                case BIDataType::QASYMM8_SIGNED:
                case BIDataType::QSYMM8_PER_CHANNEL:
                    return max_consecutive_elements_display_width_impl<int8_t>(s, reinterpret_cast<const int8_t *>(ptr), n);
                case BIDataType::U16:
                case BIDataType::QASYMM16:
                    return max_consecutive_elements_display_width_impl<uint16_t>(s, reinterpret_cast<const uint16_t *>(ptr), n);
                case BIDataType::S16:
                case BIDataType::QSYMM16:
                    return max_consecutive_elements_display_width_impl<int16_t>(s, reinterpret_cast<const int16_t *>(ptr), n);
                case BIDataType::U32:
                    return max_consecutive_elements_display_width_impl<uint32_t>(s, reinterpret_cast<const uint32_t *>(ptr), n);
                case BIDataType::S32:
                    return max_consecutive_elements_display_width_impl<int32_t>(s, reinterpret_cast<const int32_t *>(ptr), n);
                case BIDataType::U64:
                    return max_consecutive_elements_display_width_impl<uint64_t>(s, reinterpret_cast<const uint64_t *>(ptr), n);
                case BIDataType::S64:
                    return max_consecutive_elements_display_width_impl<int64_t>(s, reinterpret_cast<const int64_t *>(ptr), n);
                case BIDataType::BFLOAT16:
                    return max_consecutive_elements_display_width_impl<bfloat16>(s, reinterpret_cast<const bfloat16 *>(ptr), n);
                case BIDataType::F16:
                    return max_consecutive_elements_display_width_impl<half>(s, reinterpret_cast<const half *>(ptr), n);
                case BIDataType::F32:
                    return max_consecutive_elements_display_width_impl<float>(s, reinterpret_cast<const float *>(ptr), n);
                default:
                    BI_COMPUTE_ERROR("Undefined element size for given data type");
            }
            return 0;
        }
#endif // BI_COMPUTE_ASSERTS_ENABLED

    std::pair<unsigned int, unsigned int> scaled_dimensions(int                    width,
                                                            int                    height,
                                                            int                    kernel_width,
                                                            int                    kernel_height,
                                                            const BIPadStrideInfo &pad_stride_info,
                                                            const Size2D          &dilation)
    {
        const int dilation_x = dilation.x();
        const int dilation_y = dilation.y();
        const int pad_left   = pad_stride_info.pad_left();
        const int pad_top    = pad_stride_info.pad_top();
        const int pad_right  = pad_stride_info.pad_right();
        const int pad_bottom = pad_stride_info.pad_bottom();
        const int stride_x   = pad_stride_info.stride().first;
        const int stride_y   = pad_stride_info.stride().second;
        int       w          = 0;
        int       h          = 0;
        switch (pad_stride_info.round())
        {
            case BIDimensionRoundingType::FLOOR:
                w = static_cast<int>(std::floor(
                    (static_cast<float>(width + pad_left + pad_right - (dilation_x * (kernel_width - 1) + 1)) / stride_x) +
                    1));
                h = static_cast<int>(
                    std::floor((static_cast<float>(height + pad_top + pad_bottom - (dilation_y * (kernel_height - 1) + 1)) /
                                stride_y) +
                               1));
                break;
            case BIDimensionRoundingType::CEIL:
                w = static_cast<int>(std::ceil(
                    (static_cast<float>(width + pad_left + pad_right - (dilation_x * (kernel_width - 1) + 1)) / stride_x) +
                    1));
                h = static_cast<int>(
                    std::ceil((static_cast<float>(height + pad_top + pad_bottom - (dilation_y * (kernel_height - 1) + 1)) /
                               stride_y) +
                              1));
                break;
            default:
                BI_COMPUTE_ERROR("Unsupported rounding type");
        }

        w = std::max(1, w);
        h = std::max(1, h);
        return std::make_pair(static_cast<unsigned int>(w), static_cast<unsigned int>(h));
    }

    std::pair<int, int> scaled_dimensions_signed(
    int width, int height, int kernel_width, int kernel_height, const BIPadStrideInfo &pad_stride_info)
    {
        const int pad_left   = pad_stride_info.pad_left();
        const int pad_top    = pad_stride_info.pad_top();
        const int pad_right  = pad_stride_info.pad_right();
        const int pad_bottom = pad_stride_info.pad_bottom();
        const int stride_x   = pad_stride_info.stride().first;
        const int stride_y   = pad_stride_info.stride().second;
        int       w          = 0;
        int       h          = 0;
        switch (pad_stride_info.round())
        {
            case BIDimensionRoundingType::FLOOR:
                w = static_cast<int>(
                    std::floor((static_cast<float>(width + pad_left + pad_right - kernel_width) / stride_x) + 1));
                h = static_cast<int>(
                    std::floor((static_cast<float>(height + pad_top + pad_bottom - kernel_height) / stride_y) + 1));
                break;
            case BIDimensionRoundingType::CEIL:
                w = static_cast<int>(
                    std::ceil((static_cast<float>(width + pad_left + pad_right - kernel_width) / stride_x) + 1));
                h = static_cast<int>(
                    std::ceil((static_cast<float>(height + pad_top + pad_bottom - kernel_height) / stride_y) + 1));
                break;
            default:
                BI_COMPUTE_ERROR("Unsupported rounding type");
        }

        return std::make_pair(static_cast<int>(w), static_cast<int>(h));
    }

    std::pair<unsigned int, unsigned int> deconvolution_output_dimensions(unsigned int           in_width,
                                                                          unsigned int           in_height,
                                                                          unsigned int           kernel_width,
                                                                          unsigned int           kernel_height,
                                                                          const BIPadStrideInfo &pad_stride_info)
    {
        const unsigned int pad_left   = pad_stride_info.pad_left();
        const unsigned int pad_top    = pad_stride_info.pad_top();
        const unsigned int pad_right  = pad_stride_info.pad_right();
        const unsigned int pad_bottom = pad_stride_info.pad_bottom();
        const unsigned int stride_x   = pad_stride_info.stride().first;
        const unsigned int stride_y   = pad_stride_info.stride().second;

        BI_COMPUTE_ERROR_ON(in_width < 1 || in_height < 1);
        BI_COMPUTE_ERROR_ON(((in_width - 1) * stride_x + kernel_width) < (pad_left + pad_right));
        BI_COMPUTE_ERROR_ON(((in_height - 1) * stride_y + kernel_height) < (pad_top + pad_bottom));
        const int w = stride_x * (in_width - 1) + kernel_width - (pad_left + pad_right);
        const int h = stride_y * (in_height - 1) + kernel_height - (pad_top + pad_bottom);

        return std::make_pair<unsigned int, unsigned int>(w, h);
    }

} // namespace BatmanInfer