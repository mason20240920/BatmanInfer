//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_I_TENSOR_INFO_HPP
#define BATMANINFER_BI_I_TENSOR_INFO_HPP

#include <batman_compute/support/i_clone_able.h>
#include <data/core/bi_types.hpp>
#include <data/bi_tensor_shape.hpp>
#include <data/core/bi_strides.hpp>
#include <data/core/bi_coordinates.hpp>
#include <data/core/utils/misc/utils.hpp>

namespace BatmanInfer {
    class BIQuantizationInfo;

    class BIITensorInfo: public misc::ICloneable<BIITensorInfo>
    {
    public:
        using TensorDimsState = std::vector<int>;

        /**
         * @brief 一个在某个领域（例如工作负载）中唯一标识 ITensorInfo 的 ID。
         */
        using Id = int32_t;

        /**
         * @brief 域内的无效张量ID
         */
        static constexpr Id invalid_tensor_id = 0;

        /**
         * @brief 获取表示动态维度状态的值
         * @return 代表动态维度的值
         */
        static constexpr int32_t get_dynamic_state_value() {
            return _dynamic_dimension;
        }

        static constexpr int32_t get_static_state_value()
        {
            return _static_dimension;
        }

        virtual ~BIITensorInfo() = default;

        /**
         * @brief 将数据类型设置为指定的值。
         *
         * @warning 重置格式为 UNKNOWN
         *
         * @param data_type
         * @return
         */
        virtual BIITensorInfo &set_data_type(BIDataType data_type) = 0;

        /**
         * @brief 设置通道的数量
         * @param num_channels
         * @return
         */
        virtual BIITensorInfo &set_num_channels(int num_channels) = 0;

        /**
         * @brief 设置一个已经初始化的张量的形状。
         *
         * @warning 改变形状需要重新计算步幅，
         *          因此只有在张量尚未分配内存的情况下才可能。
         *
         * @param shape
         * @return
         */
        virtual BIITensorInfo &set_tensor_shape(const BITensorShape &shape) = 0;

        /**
         * @brief 为张量的每个维度设置状态。
         *
         * 这设置了形状每个维度的状态，以动态行为的方式使用 -1 在适当的地方。
         * 状态中的索引与形状维度索引是一对一的映射。
         * 例如，如果您想将 [?, 3, 3] 表示为动态输入，则必须将 [-1, 3, 3] 设置为状态。
         *
         * @param state
         * @return
         */
        virtual BIITensorInfo &set_tensor_dims_state(const TensorDimsState &state) = 0;

        /**
         * @brief 设置张量的量化设置（缩放和偏移）。
         * @param quantization_info
         * @return
         */
        virtual BIITensorInfo &set_quantization_info(const BIQuantizationInfo &quantization_info) = 0;

        /**
         * @brief 重置张量的填充设置。
         * @return
         */
        virtual BIITensorInfo &reset_padding() = 0;

        /**
         * @brief 将偏移量更新为第一个元素，并将步幅更新为自动计算的值。
         *
         * @note 该方法使用的填充非常保守，以便张量可以用于大多数函数。
         *
         * @return 如果步幅或第一个元素的偏移量发生变化，则为真。
         */
        virtual bool auto_padding() = 0;

        /**
         * @brief 设置张量的锁定填充标志。
         * @param flag 当张量可以映射到相机或帧缓冲时，应将其设置为真。
         * @return
         */
        virtual BIITensorInfo &set_lock_paddings(bool flag) = 0;

        /**
         * @brief 获取锁定填充标志值
         *
         * 锁定填充的用途
         *     填充（padding） 是指在张量的内存布局中，为了满足对齐需求或硬件要求，在数据的边界或中间添加额外的无效数据。
         *     当张量需要映射到外部设备（如摄像头或帧缓冲区）时，填充可能需要保持固定，不能随意更改。
         *     锁定填充 的作用是防止填充被修改，以确保张量的内存布局与外部设备的要求一致。
         * 典型场景
         *     摄像头（camera）或帧缓冲区（frame buffer）：
         *     当张量被映射到这些外部设备时，内存布局通常由设备驱动程序或硬件指定，不能随意更改。
         *     如果填充被修改，可能会导致数据对齐错误、性能下降，甚至程序崩溃。
         *     锁定填充 可以确保张量的内存布局不被意外更改，从而保证数据的正确性。
         *
         * @return
         */
        virtual bool lock_paddings() const = 0;

        /**
         * @brief 将偏移量更新为第一个元素、步幅和总大小。
         *
         * @note 此函数只能增加偏移量、步幅和总大小。
         *
         * @param padding 在XY平面周围的元素数量的填充。
         * @return
         */
        virtual bool extend_padding(const BIPaddingSize &padding) = 0;

        /**
         * @brief 返回维度的大小。
         * @param index
         * @return
         */
        virtual size_t dimension(size_t index) const = 0;

        /**
         * @brief 返回维度的大小。
         * @param dimension
         * @return
         */
        virtual size_t dimension(BIDataLayoutDimension dimension) const = 0;

        /**
         * @brief 访问张量每个维度所需的字节步幅
         * @return
         */
        virtual const BIStrides &strides_in_bytes() const = 0;

        /** Set the format of an already initialized tensor.
         *
         * @note If the data type has already been configured (i.e. not UNKNOWN) it
         * must match the new format. If data type hasn't been configured it will
         * be based on the format.
         *
         * @param[in] format Single-plane format of the tensor.
         *
         * @return Reference to this ITensorInfo object
         */
        virtual BIITensorInfo &set_format(Format format) = 0;

        /**
         * @brief 从内存分配的开始到张量的第一个元素的偏移量。
         *        这可以用于有效地访问二维张量中的元素。
         * @return
         */
        virtual size_t offset_first_element_in_bytes() const = 0;

        /**
         * @brief 从内存分配的开始到访问位置 (x, y, z ...) 的元素的字节偏移量
         * @param pos 访问元素的坐标向量。
         *            这个向量的大小必须等于张量的维度数量
         * @return
         */
        virtual int32_t offset_element_in_bytes(const BICoordinates &pos) const = 0;

        /**
         * @brief 元素大小（以字节为单位）计算为 data_size() * num_channels()
         * @return 一个元素的大小（以字节为单位）
         */
        virtual size_t element_size() const = 0;

        virtual size_t num_dimensions() const = 0;

        virtual size_t num_channels() const = 0;

        virtual const BITensorShape &tensor_shape() const = 0;

        /**
         * @brief 一个包含张量每个维度状态的向量，其中 -1 指定动态维度。
         * @return
         */
        virtual const TensorDimsState &tensor_dims_state() const = 0;

        virtual BIDataType data_type() const = 0;

        virtual Format format() const = 0;

        virtual size_t total_size() const = 0;

        virtual BIPaddingSize padding() const = 0;

        virtual bool has_padding() const = 0;

        /**
         * @brief 标志指示张量的大小是否可以更改。
         * @return
         */
        virtual bool is_resizable() const = 0;

        /**
         * @brief 标志指示张量的形状是否是动态的，这意味着它可以在内核/函数执行时发生变化。
         * @return
         */
        virtual bool is_dynamic() const = 0;

        /**
         * @brief 标志指示张量的值是否是常量，这意味着它们可以在内核/函数执行时发生变化。
         * @return
         */
        virtual bool are_values_constant() const = 0;

        virtual BIITensorInfo &set_is_resizable(bool is_resizable) = 0;

        virtual BIITensorInfo &set_are_values_constant(bool are_values_constant) = 0;

        /**
         * @brief 将内存分配的起始位置的偏移量设置为张量的第一个元素。
         * @param offset
         * @return
         */
        virtual BIITensorInfo &set_offset_first_element_in_bytes(const size_t offset) = 0;

        /**
         * @brief 张量的有效区域。有效区域中的所有元素都有定义值，即不是未定义的。
         * @return
         */
        virtual BIValidRegion valid_region() const = 0;

        /**
         * @brief 设置张量的有效区域。
         * @param valid_region
         */
        virtual void set_valid_region(const BIValidRegion &valid_region) = 0;

        virtual BIQuantizationInfo quantization_info() const = 0;

        virtual Id id() const = 0;

        virtual BIITensorInfo &set_id(BIITensorInfo::Id id) = 0;

        bool has_valid_id() const
        {
            return id() != invalid_tensor_id;
        }

        /**
        * @brief 如果信息是广播兼容的张量信息，请返回广播形状和张量的有效区域的交集。
        * 如果两个张量信息是广播兼容的，则它们的形状是广播兼容的。
        * 如果两个张量形状是广播兼容的，则对于每个维度，它们相等或其中一个为1。
        * 如果两个形状是兼容的，则广播形状中的每个维度是原始维度的最大值。
        * @tparam Infos
        * @param infos
        * @return
        */
        template <typename... Infos>
        static std::pair<BITensorShape, BIValidRegion> broadcast_shape_and_valid_region(const Infos &...infos) {
            BITensorShape bc_shape = BITensorShape::broadcast_shape(infos.tensor_shape()...);
            BIValidRegion bc_valid_region{BICoordinates(), bc_shape};

            auto broadcast_valid_region = [&bc_valid_region](const BIITensorInfo &info) {
                if (info.num_dimensions() != 0)
                {
                    for (size_t d = 0; d < bc_valid_region.shape.num_dimensions(); ++d)
                    {
                        const bool is_broadcast = (info.tensor_shape()[d] == 1);

                        const int    anchor_max = std::max(bc_valid_region.anchor[d], info.valid_region().anchor[d]);
                        const size_t valid_min  = std::min(bc_valid_region.shape[d], info.valid_region().shape[d]);

                        if (!is_broadcast || (valid_min == 0))
                        {
                            bc_valid_region.anchor.set(d, anchor_max);
                            bc_valid_region.shape.set(d, valid_min);
                        }
                    }
                }
            };

            misc::utility::for_each(broadcast_valid_region, infos...);

            return std::pair<BITensorShape, BIValidRegion>(bc_shape, bc_valid_region);
        }
    private:
        /**
         * @brief 动态维度
         */
        static constexpr int32_t _dynamic_dimension = -1;
        /**
         * @brief 静态维度
         */
        static constexpr int32_t _static_dimension = 0;
    };
}

#endif //BATMANINFER_BI_I_TENSOR_INFO_HPP
