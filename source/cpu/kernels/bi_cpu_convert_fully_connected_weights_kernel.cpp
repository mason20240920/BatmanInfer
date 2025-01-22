//
// Created by Mason on 2025/1/22.
//

#include <cpu/kernels/bi_cpu_convert_fully_connected_weights_kernel.hpp>

#include <data/core/bi_helpers.hpp>
#include <data/core/bi_types.hpp>

#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            void BICpuConvertFullyConnectedWeightsKernel::configure(const BatmanInfer::BIITensorInfo *src,
                                                                    BatmanInfer::BIITensorInfo *dst,
                                                                    const BatmanInfer::BITensorShape &original_input_shape,
                                                                    BatmanInfer::BIDataLayout data_layout) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);

                // 输出张量如果没有初始化就自动初始化
                auto_init_if_empty(*dst, *src->clone());

                BI_COMPUTE_ERROR_THROW_ON(
                        BICpuConvertFullyConnectedWeightsKernel::validate(src, dst, original_input_shape, data_layout));

                const BIDataLayout input_data_layout = (data_layout == BIDataLayout::NCHW) ? BIDataLayout::NHWC
                                                                                           : BIDataLayout::NCHW;

                const int width_idx = get_data_layout_dimension_index(input_data_layout, BIDataLayoutDimension::WIDTH);
                const int height_idx = get_data_layout_dimension_index(input_data_layout,
                                                                       BIDataLayoutDimension::HEIGHT);
                const int channel_idx = get_data_layout_dimension_index(input_data_layout,
                                                                        BIDataLayoutDimension::CHANNEL);

                const unsigned int num_elems_per_input_plane =
                        original_input_shape[width_idx] * original_input_shape[height_idx];
                const unsigned int num_channels = original_input_shape[channel_idx];

                _factor1 = (data_layout == BIDataLayout::NCHW) ? num_elems_per_input_plane : num_channels;
                _factor2 = (data_layout == BIDataLayout::NCHW) ? num_channels : num_elems_per_input_plane;

                // Configure kernel window
                BIWindow win = calculate_max_window(*src, BISteps());
                BIICpuKernel::configure(win);
            }

            BIStatus BICpuConvertFullyConnectedWeightsKernel::validate(const BatmanInfer::BIITensorInfo *src,
                                                                       const BatmanInfer::BIITensorInfo *dst,
                                                                       const BatmanInfer::BITensorShape &original_input_shape,
                                                                       BatmanInfer::BIDataLayout data_layout) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
                BI_COMPUTE_RETURN_ERROR_ON(src->data_type() == BIDataType::UNKNOWN);
                BI_COMPUTE_RETURN_ERROR_ON(src->num_dimensions() != 2);
                BI_COMPUTE_RETURN_ERROR_ON(src->dimension(1) != original_input_shape.total_size_lower(3));
                BI_COMPUTE_RETURN_ERROR_ON(data_layout == BIDataLayout::UNKNOWN);

                // Checks performed when dst is configured
                if ((dst != nullptr) && (dst->total_size() != 0)) {
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
                }

                return BIStatus{};
            }

            void BICpuConvertFullyConnectedWeightsKernel::run_op(BatmanInfer::BIITensorPack &tensors,
                                                                 const BatmanInfer::BIWindow &window,
                                                                 const BatmanInfer::ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                const auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                // X的步长
                const unsigned int dst_stride_x = dst->info()->strides_in_bytes().x();
                // Y方向的步长
                const unsigned int dst_stride_y = dst->info()->strides_in_bytes().y();
                // 每个元素的字节数（以字节为单位，例如 float 是 4 字节）
                const unsigned int element_size = src->info()->element_size();

                BIIterator input(src, window);
                BIIterator output(dst, window);

                // 利用factor1和factor2进行映射, 从NCHW转为NHWC
                execute_window_loop(window, [&](const BICoordinates &id) {
                    memcpy(output.ptr() + id.x() * dst_stride_x +
                           (id.y() % _factor1 * _factor2 + id.y() / _factor1) * dst_stride_y,
                           input.ptr(), element_size);
                }, input);
            }

            const char *BICpuConvertFullyConnectedWeightsKernel::name() const {
                return "BICpuConvertFullyConnectedWeightsKernel";
            }

        }
    }
}