//
// Created by Mason on 2024/12/31.
//

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_window.hpp>
#include <data/core/bi_helpers.hpp>
#include <data/core/utils/data_type_utils.hpp>
#include <data/core/bi_utils.hpp>

#include <cstring>
#include <cstdint>
#include <limits>
#include <sstream>

namespace BatmanInfer {
    void BIITensor::copy_from(const BatmanInfer::BIITensor &src) {
        if (&src == this)
            return;

        const BIITensorInfo *src_info = src.info();
        BIITensorInfo       *dst_info = this->info();

        // 目标张量维度小于源维度
        BI_COMPUTE_ERROR_ON(src_info->num_dimensions() > dst_info->num_dimensions());
        BI_COMPUTE_ERROR_ON(src_info->num_channels() != dst_info->num_channels());
        BI_COMPUTE_ERROR_ON(src_info->element_size() != dst_info->element_size());

        for (size_t d = 0; d < src_info->num_dimensions(); d++)
            BI_COMPUTE_ERROR_ON(src_info->dimension(d) > dst_info->dimension(d));

        // 拷贝关于可用区间的信息
        dst_info->set_valid_region(src_info->valid_region());

        BIWindow win_src;
        win_src.use_tensor_dimensions(src_info->tensor_shape(), BIWindow::DimY);
        BIWindow win_dst;
        win_dst.use_tensor_dimensions(dst_info->tensor_shape(), BIWindow::DimY);

        BIIterator src_it(&src, win_src);
        BIIterator dst_it(this, win_dst);

        const size_t line_size = src_info->element_size() * src.info()->dimension(0);

        execute_window_loop(win_src, [&](const BICoordinates &) { memcpy(dst_it.ptr(), src_it.ptr(), line_size); },
                            src_it, dst_it);
    }

#ifdef BI_COMPUTE_ASSERTS_ENABLED

    void BIITensor::print(std::ostream &s, BIIOFormatInfo io_fmt) const {
        BI_COMPUTE_ERROR_ON(this->buffer() == nullptr);

        const BIDataType    dt           = this->info()->data_type();
        const size_t        slices2D     = this->info()->tensor_shape().total_size_upper(2);
        const BIStrides     strides      = this->info()->strides_in_bytes();
        const BIPaddingSize padding      = this->info()->padding();
        const size_t        num_channels = this->info()->num_channels();
        // 使用 std::ostringstream 保存流状态
        std::ostringstream  stream_status;
        stream_status.copyfmt(s);

        // Set precision
        if (is_data_type_float(dt) && (io_fmt.precision_type != BIIOFormatInfo::PrecisionType::Default)) {
            int precision = io_fmt.precision;
            if (io_fmt.precision_type == BIIOFormatInfo::PrecisionType::Full) {
                precision = std::numeric_limits<float>().max_digits10;
            }
            s.precision(precision);
        }

        // Define region to print
        size_t print_width  = 0;
        size_t print_height = 0;
        int    start_offset = 0;
        switch (io_fmt.print_region) {
            case BIIOFormatInfo::PrintRegion::NoPadding:
                print_width  = this->info()->dimension(0);
                print_height = this->info()->dimension(1);
                start_offset = this->info()->offset_first_element_in_bytes();
                break;
            case BIIOFormatInfo::PrintRegion::ValidRegion:
                print_width  = this->info()->valid_region().shape.x();
                print_height = this->info()->valid_region().shape.y();
                start_offset = this->info()->offset_element_in_bytes(
                        BICoordinates(this->info()->valid_region().anchor.x(), this->info()->valid_region().anchor.y()));
                break;
            case BIIOFormatInfo::PrintRegion::Full:
                print_width  = padding.left + this->info()->dimension(0) + padding.right;
                print_height = padding.top + this->info()->dimension(1) + padding.bottom;
                start_offset =
                        static_cast<int>(this->info()->offset_first_element_in_bytes()) - padding.top * strides[1] -
                        padding.left * strides[0];
                break;
            default:
                break;
        }

        print_width = print_width * num_channels;

        // Set pointer to start
        const uint8_t *ptr = this->buffer() + start_offset;

        // Start printing
        for (size_t i = 0; i < slices2D; ++i) {
            // Find max_width of elements in slice to align columns
            int max_element_width = 0;
            if (io_fmt.align_columns) {
                size_t      offset = i * strides[2];
                for (size_t h      = 0; h < print_height; ++h) {
                    max_element_width = std::max<int>(
                            max_element_width,
                            max_consecutive_elements_display_width(s, dt, ptr + offset, print_width));
                    offset += strides[1];
                }
            }

            // Print slice
            {
                size_t      offset = i * strides[2];
                for (size_t h      = 0; h < print_height; ++h) {
                    print_consecutive_elements(s, dt, ptr + offset, print_width, max_element_width,
                                               io_fmt.element_delim);
                    offset += strides[1];
                    s << io_fmt.row_delim;
                }
                s << io_fmt.row_delim;
            }
        }

        // Restore output stream flags
        s.copyfmt(stream_status);
    }

#endif

    bool BIITensor::is_used() const
    {
        return _is_used;
    }

    void BIITensor::mark_as_unused() const
    {
        _is_used = false;
    }

    void BIITensor::mark_as_used() const
    {
        _is_used = true;
    }
}