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
#endif
}