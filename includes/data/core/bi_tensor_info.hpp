//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_TENSOR_INFO_HPP
#define BATMANINFER_BI_TENSOR_INFO_HPP

#include <data/core/bi_i_tensor_info.hpp>
#include <data/core/quantization_info.hpp>
#include <data/core/utils/data_type_utils.hpp>
#include <data/core/bi_helpers.hpp>

namespace BatmanInfer {

    /**
     * @brief 存储张量元数据
     */
    class BITensorInfo final : public BIITensorInfo {
    public:
        BITensorInfo();

        ~BITensorInfo() override = default;

        BITensorInfo(const BITensorInfo &);

        BITensorInfo(const BIITensorInfo &info);

        BITensorInfo &operator=(const BITensorInfo &) = default;

        BITensorInfo(BITensorInfo &&) = default;

        BITensorInfo &operator=(BITensorInfo &&) = default;

        /** 初始化函数 **/
        explicit BITensorInfo(Format format);

        explicit BITensorInfo(unsigned int width, unsigned int height, Format format);

        explicit BITensorInfo(const BITensorShape &tensor_shape, Format format);

        explicit BITensorInfo(const BITensorShape &tensor_shape,
                              size_t num_channels,
                              BIDataType data_type);

        explicit BITensorInfo(size_t num_channels, BIDataType data_type);

        explicit BITensorInfo(const BITensorShape &tensor_shape,
                              size_t num_channels,
                              BIDataType data_type,
                              BIQuantizationInfo quantization_info);

        /** Initialize the tensor info with just a format.
          *
          * Can be used for automatic derivation of the shape by the function.
          *
          * @param[in] format Single plane format of the tensor.
          */
        void init(Format format);

        /** Initialize the metadata structure with the given parameters
         *
         * @param[in] tensor_shape Size for each dimension of the tensor in number of elements.
         * @param[in] format       Single plane format of the tensor.
         */
        void init(const BITensorShape &tensor_shape, Format format);

        /** Initialize the metadata structure with the given parameters
         *
         * @param[in] tensor_shape                  Size for each dimension of the tensor in number of elements.
         * @param[in] format                        Single plane format of the tensor.
         * @param[in] strides_in_bytes              Stride in bytes for accessing each dimension of the tensor.
         * @param[in] offset_first_element_in_bytes Offset in bytes from the beginning of memory allocation to access the first element.
         * @param[in] total_size_in_bytes           Size in bytes of the memory allocation (including the offset to the first element).
         */
        void init(const BITensorShape &tensor_shape,
                  Format format,
                  const BIStrides &strides_in_bytes,
                  size_t offset_first_element_in_bytes,
                  size_t total_size_in_bytes);

        /** Initialize the tensor info with just a format.
         *
         * Can be used for automatic derivation of the shape by the function.
         *
         * @param[in] num_channels Desired number of channels for each tensor element.
         * @param[in] data_type    Data type to use for each tensor element.
         */
        void init(size_t num_channels, BIDataType data_type);

        /** Initialize the metadata structure with the given parameters
         *
         * @param[in] tensor_shape Size for each dimension of the tensor in number of elements.
         * @param[in] num_channels Desired number of channels for each tensor element.
         * @param[in] data_type    Data type to use for each tensor element.
         */
        void init(const BITensorShape &tensor_shape, size_t num_channels, BIDataType data_type);

        /** Initialize the metadata structure with the given parameters
         *
         * @param[in] tensor_shape                  Size for each dimension of the tensor in number of elements.
         * @param[in] num_channels                  Desired number of channels for each tensor element.
         * @param[in] data_type                     Data type to use for each tensor element.
         * @param[in] strides_in_bytes              Stride in bytes for accessing each dimension of the tensor.
         * @param[in] offset_first_element_in_bytes Offset in bytes from the beginning of memory allocation to access the first element.
         * @param[in] total_size_in_bytes           Size in bytes of the memory allocation (including the offset to the first element).
         */
        void init(const BITensorShape &tensor_shape,
                  size_t num_channels,
                  BIDataType data_type,
                  const BIStrides &strides_in_bytes,
                  size_t offset_first_element_in_bytes,
                  size_t total_size_in_bytes);

        /** Initialize the metadata structure for the given tensor shape and single-plane format, (Padding is automatically calculated)
         *
         * @note The padding used by this method is really conservative so that the tensor can be used for most functions.
         *
         * @param[in] tensor_shape It specifies the size for each dimension of the tensor in number of elements
         * @param[in] format       Single plane format of the image.
         *
         * @return Total allocation size including padding in bytes.
         */
        size_t init_auto_padding(const BITensorShape &tensor_shape, Format format);

        /** Initialize the metadata structure for the given tensor shape, number of channels and
         *  data type. (Padding is automatically calculated)
         *
         * @note The padding used by this method is really conservative so that the tensor can be used for most functions.
         *
         * @param[in] tensor_shape It specifies the size for each dimension of the tensor in number of elements
         * @param[in] num_channels It indicates the number of channels for each tensor element
         * @param[in] data_type    Data type to use for each tensor element
         *
         * @return Total allocation size including padding in bytes.
         */
        size_t init_auto_padding(const BITensorShape &tensor_shape, size_t num_channels, BIDataType data_type);

        // Inherited methods overridden:
        std::unique_ptr<BIITensorInfo> clone() const override;

        BIITensorInfo &set_data_type(BIDataType data_type) override;

        BIITensorInfo &set_num_channels(int num_channels) override;

        BIITensorInfo &set_format(Format format) override;

        BIITensorInfo &set_tensor_shape(const BITensorShape &shape) override;

        BIITensorInfo &set_tensor_dims_state(const TensorDimsState &state) override;

        BIITensorInfo &set_quantization_info(const BIQuantizationInfo &quantization_info) override;

        BIITensorInfo &reset_padding() override;

        bool auto_padding() override;

        BIITensorInfo &set_lock_paddings(bool flag) override;

        bool lock_paddings() const override;

        bool extend_padding(const BIPaddingSize &padding) override;

        size_t dimension(size_t index) const override {
            return _tensor_shape[index];
        }

        size_t dimension(BIDataLayoutDimension dimension) const override {
            return get_data_layout_dimension_index(dimension);
        }

        const BIStrides &strides_in_bytes() const override {
            return _strides_in_bytes;
        }

        size_t offset_first_element_in_bytes() const override {
            return _offset_first_element_in_bytes;
        }

        int32_t offset_element_in_bytes(const BICoordinates &pos) const override;

        size_t element_size() const override {
            return data_size_from_type(_data_type) * _num_channels;
        }

        size_t num_dimensions() const override {
            return _tensor_shape.num_dimensions();
        }

        size_t num_channels() const override {
            return _num_channels;
        }

        const BITensorShape &tensor_shape() const override {
            return _tensor_shape;
        }

        const TensorDimsState &tensor_dims_state() const override {
            return _dims_state;
        }

        BIDataType data_type() const override {
            return _data_type;
        }

        Format format() const override {
            return _format;
        }

        size_t total_size() const override {
            return _total_size;
        }

        BIPaddingSize padding() const override {
            return _padding;
        }

        bool has_padding() const override {
            return !_padding.empty();
        }

        bool is_resizable() const override {
            return _is_resizable;
        }

        bool is_dynamic() const override {
            return std::find(std::cbegin(_dims_state), std::cend(_dims_state), get_dynamic_state_value()) !=
                   std::cend(_dims_state);
        }

        bool are_values_constant() const override {
            return _are_values_constant;
        }

        BIITensorInfo &set_is_resizable(bool is_resizable) override {
            _is_resizable = is_resizable;
            return *this;
        }

        BIValidRegion valid_region() const override {
            return _valid_region;
        }

        void set_valid_region(const BIValidRegion &valid_region) override {
            _valid_region = valid_region;
        }

        BIQuantizationInfo quantization_info() const override {
            return _quantization_info;
        }

        BIITensorInfo &set_are_values_constant(bool are_values_constant) override {
            _are_values_constant = are_values_constant;
            return *this;
        }

        BIITensorInfo::Id id() const override {
            return _id;
        }

        BIITensorInfo &set_id(BIITensorInfo::Id id) override {
            _id = id;
            return *this;
        }

        BIITensorInfo &set_offset_first_element_in_bytes(const size_t offset) override {
            _offset_first_element_in_bytes = offset;
            return *this;
        }

        inline friend bool operator==(const BITensorInfo &lhs, const BITensorInfo &rhs);

    private:

        /**
        * @brief 计算在 XY 平面周围指定填充后产生的步幅、偏移量和总大小。
        *
        * @note 当解释返回值中的 required_strides 时，仅对应张量维度的值是有效的。
        *       例如，对于一维张量，仅一维的 required_strides 是有效的；
        *       对于二维张量，有效值为前两维的 required_strides，以此类推。
        *
        * @param[in] padding 在 XY 平面周围按元素数指定的填充。
        *
        * @return 返回一个包含以下内容的元组：
        *         - Strides（步幅）：张量在每个维度上的步幅。
        *         - size_t（偏移量）：张量的起始偏移量。
        *         - size_t（总大小）：张量的总内存大小（包括填充）。
        */

        std::tuple<BIStrides, size_t, size_t> calculate_padding_requirements(const BIPaddingSize &padding);

        /**
         * @brief 张量的总体积
         */
        size_t _total_size;
        /**
         * @brief 第一个元素的字节偏差
         */
        size_t _offset_first_element_in_bytes;
        BIStrides _strides_in_bytes;
        size_t _num_channels;
        BITensorShape _tensor_shape;
        TensorDimsState _dims_state;
        BIDataType _data_type;
        Format _format;
        bool _is_resizable;
        BIValidRegion _valid_region;
        BIPaddingSize _padding;
        BIQuantizationInfo _quantization_info;
        bool _are_values_constant;
        BIITensorInfo::Id _id;
        bool _lock_paddings;
    };

    /** Check whether two tensor info are equal.
 *
 * @param[in] lhs LHS tensor info.
 * @param[in] rhs RHS tensor info.
 *
 * @return True if the given tensor infos are the same.
 */
    inline bool operator==(const BITensorInfo &lhs, const BITensorInfo &rhs) {
        return (lhs._total_size == rhs._total_size) &&
               (lhs._offset_first_element_in_bytes == rhs._offset_first_element_in_bytes) &&
               (lhs._strides_in_bytes == rhs._strides_in_bytes) && (lhs._num_channels == rhs._num_channels) &&
               (lhs._tensor_shape == rhs._tensor_shape) && (lhs._dims_state == rhs._dims_state) &&
               (lhs._data_type == rhs._data_type) && (lhs._format == rhs._format) &&
               (lhs._is_resizable == rhs._is_resizable) && (lhs._valid_region == rhs._valid_region) &&
               (lhs._padding == rhs._padding) && (lhs._quantization_info == rhs._quantization_info) &&
               (lhs._are_values_constant == rhs._are_values_constant) &&
               (lhs._id == rhs._id);
    }
}

#endif //BATMANINFER_BI_TENSOR_INFO_HPP
