//
// Created by Mason on 2024/12/26.
//

#include <data/core/bi_tensor_info.hpp>
#include <data/core/utils/data_type_utils.hpp>
#include <data/core/utils/format_utils.hpp>
#include <data/core/utils/helpers/bi_utils.hpp>

namespace BatmanInfer {
    BITensorInfo::BITensorInfo() :
            _total_size(0),
            _offset_first_element_in_bytes(0),
            _strides_in_bytes(),
            _num_channels(0),
            _tensor_shape(),
            _dims_state(),
            _data_type(BIDataType::UNKNOWN),
            _format(Format::UNKNOWN),
            _is_resizable{true},
            _valid_region{BICoordinates(), _tensor_shape},
            _padding{0},
            _quantization_info(),
            _are_values_constant(true),
            _id(invalid_tensor_id),
            _lock_paddings(false){

    }

    BITensorInfo::BITensorInfo(const BIITensorInfo &info) : BITensorInfo()
    {
        _total_size                    = info.total_size();
        _offset_first_element_in_bytes = info.offset_first_element_in_bytes();
        _strides_in_bytes              = info.strides_in_bytes();
        _num_channels                  = info.num_channels();
        _tensor_shape                  = info.tensor_shape();
        _dims_state                    = info.tensor_dims_state();
        _data_type                     = info.data_type();
        _format                        = info.format();
        _is_resizable                  = info.is_resizable();
        _valid_region                  = info.valid_region();
        _padding                       = info.padding();
        _quantization_info             = info.quantization_info();
        _are_values_constant           = info.are_values_constant();
        _id                            = info.id();
        _lock_paddings                 = info.lock_paddings();
    }

    BITensorInfo::BITensorInfo(const BITensorInfo &info) : BITensorInfo()
    {
        _total_size                    = info.total_size();
        _offset_first_element_in_bytes = info.offset_first_element_in_bytes();
        _strides_in_bytes              = info.strides_in_bytes();
        _num_channels                  = info.num_channels();
        _tensor_shape                  = info.tensor_shape();
        _dims_state                    = info.tensor_dims_state();
        _data_type                     = info.data_type();
        _format                        = info.format();
        _is_resizable                  = info.is_resizable();
        _valid_region                  = info.valid_region();
        _padding                       = info.padding();
        _quantization_info             = info.quantization_info();
        _are_values_constant           = info.are_values_constant();
        _id                            = info.id();
        _lock_paddings                 = false;
    }

    BITensorInfo::BITensorInfo(Format format) : BITensorInfo(BITensorShape(), format)
    {
    }

    BITensorInfo::BITensorInfo(unsigned int width, unsigned int height, Format format)
            : BITensorInfo(BITensorShape(width, height), format)
    {
    }

    BITensorInfo::BITensorInfo(const BITensorShape &tensor_shape, Format format) : BITensorInfo()
    {
        init(tensor_shape, format);
    }

    BITensorInfo::BITensorInfo(size_t num_channels, BIDataType data_type) : BITensorInfo()
    {
        init(BITensorShape(), num_channels, data_type);
    }

    BITensorInfo::BITensorInfo(const BITensorShape &tensor_shape,
                               size_t             num_channels,
                               BIDataType           data_type,
                               BIQuantizationInfo   quantization_info)
            : BITensorInfo()
    {
        init(tensor_shape, num_channels, data_type);
        _quantization_info = std::move(quantization_info);
    }

    void BITensorInfo::init(Format format)
    {
        init(BITensorShape(), format);
    }

    void BITensorInfo::init(const BITensorShape &tensor_shape, Format format)
    {
        size_t         num_channels = num_channels_from_format(format);
        const BIDataType type         = data_type_from_format(format);

        init(tensor_shape, num_channels, type);

        _format = format;
    }

    void BITensorInfo::init(const BITensorShape &tensor_shape,
                          Format             format,
                          const BIStrides     &strides_in_bytes,
                          size_t             offset_first_element_in_bytes,
                          size_t             total_size_in_bytes)
    {
        size_t         num_channels = num_channels_from_format(format);
        const BIDataType type         = data_type_from_format(format);

        init(tensor_shape, num_channels, type, strides_in_bytes, offset_first_element_in_bytes, total_size_in_bytes);

        _format = format;
    }

    void BITensorInfo::init(size_t num_channels, BIDataType data_type)
    {
        init(BITensorShape(), num_channels, data_type);
    }

    void BITensorInfo::init(const BITensorShape &tensor_shape,
                            size_t num_channels,
                            BIDataType data_type)
    {
        ARM_COMPUTE_ERROR_ON(num_channels == 0);

        _data_type    = data_type;
        _num_channels = num_channels;
        _format       = Format::UNKNOWN;

        set_tensor_shape(tensor_shape);
    }

    void BITensorInfo::init(const BITensorShape &tensor_shape,
                          size_t             num_channels,
                            BIDataType           data_type,
                          const BIStrides     &strides_in_bytes,
                          size_t             offset_first_element_in_bytes,
                          size_t             total_size_in_bytes)
    {
        ARM_COMPUTE_ERROR_ON(num_channels == 0);

        _data_type                     = data_type;
        _num_channels                  = num_channels;
        _format                        = Format::UNKNOWN;
        _tensor_shape                  = tensor_shape;
        _offset_first_element_in_bytes = offset_first_element_in_bytes;
        _strides_in_bytes              = strides_in_bytes;
        _total_size                    = total_size_in_bytes;

        _valid_region = BIValidRegion{BICoordinates(), _tensor_shape};
    }

    size_t BITensorInfo::init_auto_padding(const BITensorShape &tensor_shape, Format format)
    {
        const size_t   num_channels = num_channels_from_format(format);
        const BIDataType type         = data_type_from_format(format);
        size_t         total_size   = init_auto_padding(tensor_shape, num_channels, type);

        _format = format;

        return total_size;
    }

    size_t BITensorInfo::init_auto_padding(const BITensorShape &tensor_shape, size_t num_channels, BIDataType data_type)
    {
        ARM_COMPUTE_ERROR_ON(num_channels == 0);

        _data_type    = data_type;
        _num_channels = num_channels;
        _format       = Format::UNKNOWN;
        _tensor_shape = tensor_shape;

        _valid_region = BIValidRegion{BICoordinates(), _tensor_shape};

        auto_padding();

        return _total_size;
    }

    bool BITensorInfo::auto_padding()
    {
        ARM_COMPUTE_ERROR_ON(!_is_resizable);

        // 一些内核同时计算32个元素，最坏情况下它们 （例如 SIMD 矢量化操作）
        // 将在最后一个元素之后读取32个值。
        const size_t extra_pad_x = _tensor_shape.num_dimensions() < 1 ? 0 : 32;
        // 基础填充通常是为了对齐内存访问（如 4 字节对齐）。这在高性能计算中非常重要，因为未对齐的内存访问会降低性能。
        const size_t pad_x       = _tensor_shape.num_dimensions() < 1 ? 0 : 4;
        const size_t pad_y       = _tensor_shape.num_dimensions() < 2 ? 0 : 4;

        return extend_padding(BIPaddingSize(pad_y, pad_x + extra_pad_x, pad_y, pad_x));
    }

    std::tuple<BIStrides, size_t, size_t> BITensorInfo::calculate_padding_requirements(const BIPaddingSize &padding)
    {
        // Calculate resulting stride for the X, Y and Z dimension
        const size_t stride_x = element_size();
        const size_t stride_y = (padding.left + _tensor_shape[0] + padding.right) * stride_x;
        const size_t stride_z = (padding.top + _tensor_shape[1] + padding.bottom) * stride_y;

        BIStrides      required_strides;
        size_t       required_total_size           = 0;
        const size_t required_offset_first_element = padding.left * stride_x + padding.top * stride_y;

        switch (_tensor_shape.num_dimensions())
        {
            case 0:
            {
                if (_tensor_shape.total_size() > 0)
                {
                    required_strides    = BIStrides(stride_x, stride_x);
                    required_total_size = stride_z;
                }
                break;
            }
            case 1:
            case 2: {
                required_strides    = compute_strides(*this, stride_x, stride_y);
                required_total_size = stride_z;
                break;
            }
            default:
            {
                required_strides = compute_strides(*this, stride_x, stride_y, stride_z);

                const unsigned int idx_last_dimension = _tensor_shape.num_dimensions() - 1;

                required_total_size =
                        static_cast<size_t>(_tensor_shape[idx_last_dimension]) * required_strides[idx_last_dimension];
                break;
            }
        }

        return std::make_tuple(required_strides, required_offset_first_element, required_total_size);
    }

    BIITensorInfo &BITensorInfo::set_lock_paddings(bool flag)
    {
        _lock_paddings = flag;
        return *this;
    }

    bool BITensorInfo::lock_paddings() const
    {
        return _lock_paddings;
    }

    bool BITensorInfo::extend_padding(const BIPaddingSize &padding)
    {
        ARM_COMPUTE_ERROR_ON(_lock_paddings);
        ARM_COMPUTE_ERROR_ON(!_is_resizable);

        bool updated = false;

        if (padding.top > _padding.top)
        {
            _padding.top = padding.top;
            updated      = true;
        }

        if (padding.right > _padding.right)
        {
            _padding.right = padding.right;
            updated        = true;
        }

        if (padding.bottom > _padding.bottom)
        {
            _padding.bottom = padding.bottom;
            updated         = true;
        }

        if (padding.left > _padding.left)
        {
            _padding.left = padding.left;
            updated       = true;
        }

        std::tie(_strides_in_bytes, _offset_first_element_in_bytes, _total_size) = calculate_padding_requirements(_padding);

        return updated;
    }

    std::unique_ptr<BIITensorInfo> BITensorInfo::clone() const
    {
        return std::make_unique<BITensorInfo>(*this);
    }

    BIITensorInfo &BITensorInfo::set_data_type(BIDataType data_type)
    {
        _data_type = data_type;
        _format    = Format::UNKNOWN;
        return set_tensor_shape(tensor_shape()); // Force total size and strides to update
    }

    BIITensorInfo &BITensorInfo::set_num_channels(int num_channels)
    {
        _num_channels = num_channels;
        _format       = Format::UNKNOWN;
        return *this;
    }

    BIITensorInfo &BITensorInfo::set_format(Format format)
    {
        _format = format;

        if (_data_type == BIDataType::UNKNOWN)
        {
            _num_channels = num_channels_from_format(format);
            _data_type    = data_type_from_format(format);
        }
        else
        {
            ARM_COMPUTE_ERROR_ON(num_channels_from_format(format) != _num_channels);
            ARM_COMPUTE_ERROR_ON(data_type_from_format(format) != _data_type);
        }
        return *this;
    }

    BIITensorInfo &BITensorInfo::set_tensor_shape(const BITensorShape &shape)
    {
        _tensor_shape                  = shape;
        _offset_first_element_in_bytes = 0;
        _strides_in_bytes              = compute_strides(*this);

        if (_tensor_shape.num_dimensions() == 0)
        {
            _total_size = _strides_in_bytes[0];
        }
        else
        {
            const unsigned int idx_last_dimension = _tensor_shape.num_dimensions() - 1;
            _total_size = static_cast<size_t>(_tensor_shape[idx_last_dimension]) * _strides_in_bytes[idx_last_dimension];
        }

        std::tie(_strides_in_bytes, _offset_first_element_in_bytes, _total_size) = calculate_padding_requirements(_padding);

        _valid_region = BIValidRegion{BICoordinates(), _tensor_shape};
        return *this;
    }

    BIITensorInfo &BITensorInfo::set_tensor_dims_state(const TensorDimsState &state)
    {
        _dims_state = state;
        return *this;
    }

    BIITensorInfo &BITensorInfo::set_quantization_info(const BIQuantizationInfo &quantization_info)
    {
        _quantization_info = quantization_info;
        return *this;
    }

    BIITensorInfo &BITensorInfo::reset_padding()
    {
        _padding = BIPaddingSize();
        if (((_format != Format::UNKNOWN) || (_data_type != BIDataType::UNKNOWN)) && _total_size != 0)
        {
            std::tie(_strides_in_bytes, _offset_first_element_in_bytes, _total_size) =
                    calculate_padding_requirements(_padding);
        }
        return *this;
    }

    int32_t BITensorInfo::offset_element_in_bytes(const BICoordinates &pos) const
    {

        auto offset = static_cast<int32_t>(_offset_first_element_in_bytes);

        for (size_t i = 0; i < _tensor_shape.num_dimensions(); ++i)
        {
            offset += pos[i] * _strides_in_bytes[i];
        }

        return offset;
    }

}