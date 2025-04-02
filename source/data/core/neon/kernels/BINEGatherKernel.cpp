//
// Created by Mason on 2025/4/1.
//

#include <data/core/neon/kernels/BINEGatherKernel.hpp>

#include <data/core/bi_coordinates.hpp>
#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>

namespace BatmanInfer {
    namespace {
        BIStatus validate_arguments(const BIITensorInfo *input,
                                    const BIITensorInfo *indices,
                                    const BIITensorInfo *output,
                                    int axis) {
            BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, indices, output);
            BI_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);

            if (axis < 0) {
                axis += input->num_dimensions();
            }

            BI_COMPUTE_RETURN_ERROR_ON(0 > axis || axis >= static_cast<int32_t>(input->num_dimensions()));
            BI_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() + indices->num_dimensions() - 1 >
                BICoordinates::num_max_dimensions);
            BI_COMPUTE_RETURN_ERROR_ON(input->data_type() == BIDataType::UNKNOWN);

            if (output->total_size() != 0) {
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
                BITensorShape output_shape = BatmanInfer::misc::shape_calculator::compute_gather_shape(
                    input->tensor_shape(), indices->tensor_shape(), axis);
                BI_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() != output->tensor_shape().total_size());
            }

            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, BIDataType::U32, BIDataType::S32);

            return BatmanInfer::BIStatus{};
        }
    }

    BINEGatherKernel::BINEGatherKernel()
        : _input{}, _indices{}, _axis{}, _output{}, _func{}, _src_it_strides{}, _idx_it_strides{} {
    }

    template<typename TIndex>
    void BINEGatherKernel::gather_common(const BIWindow &window, const ThreadInfo &info) {
        BI_COMPUTE_UNUSED(info);

        auto dst_win = window; // 获取window进行运行

        const auto src_info = _input->info(); // 输入weights的张量信息
        const auto idx_info = _indices->info(); // indices的张量信息
        const auto dst_info = _output->info(); // 输出output的张量信息

        const auto num_dims = dst_info->num_dimensions(); // 输出张量的维度数
        const auto chunk_stride = src_info->strides_in_bytes()[_axis]; // 在哪个维度上进行处理

        const auto window_start_x = window.x().start(); // 遍历开始的点
        const auto window_end_x = window.x().end(); // 遍历结束的点
        auto window_size_x = src_info->element_size(); // 窗口大小

        const auto idx_limit = static_cast<TIndex>(src_info->tensor_shape()[_axis]); // 输出的张量信息

        if (_axis != 0) {
            // 轴不为0, 则修改windows的维度
            dst_win.set(0, BIWindow::BIDimension(window_start_x, window_start_x + 1, 1));
            window_size_x *= window_end_x - window_start_x;
        }

        // Compute source and index tensors window based on the output window.
        auto src_win = dst_win;
        BIWindow idx_win;

        for (size_t i = 0; i < idx_info->num_dimensions(); ++i) {
            src_win.set(_axis + i, BIWindow::BIDimension(0, 1, 1));
            idx_win.set(_axis + i, window[_axis + i]);
        }

        // Use the custom strides to access all three tensors using the same loop.
        BIIterator src_it(num_dims, _src_it_strides, _input->buffer(), src_info->offset_first_element_in_bytes(),
                          src_win);
        BIIterator idx_it(num_dims, _idx_it_strides, _indices->buffer(), idx_info->offset_first_element_in_bytes(),
                          idx_win);
        BIIterator dst_it(num_dims, dst_info->strides_in_bytes(), _output->buffer(),
                          dst_info->offset_first_element_in_bytes(), dst_win);

        execute_window_loop(
            dst_win,
            [&](const BICoordinates &) {
                const auto idx = *reinterpret_cast<const TIndex *>(idx_it.ptr());

                if (idx >= 0 && idx < idx_limit) {
                    const auto src_ptr = src_it.ptr() + idx * chunk_stride;

                    std::copy_n(src_ptr, window_size_x, dst_it.ptr());
                } else {
                    std::fill_n(dst_it.ptr(), window_size_x, 0);
                }
            },
            src_it, idx_it, dst_it);
    }

    void BINEGatherKernel::dynamic_configure(const BIITensor *indices, BIITensor *output) {
        _indices = indices;
        _output = output;
        // Create window: (Window进行动态修改)
        BIWindow win = BIIKernel::window();
        dynamic_calculate_max_window(*output->info(), win);
        BIIKernel::dynamic_configure(win);

        // 更新索引的indices信息
        const auto indices_info = indices->info();
        const auto &indices_strides = indices_info->strides_in_bytes();
        const auto indices_num_dims = indices_info->num_dimensions();

        const auto input_info = _input->info();
        const auto &input_strides = input_info->strides_in_bytes();

        size_t dim_no = static_cast<size_t>(_axis);
        for (; dim_no < static_cast<size_t>(_axis) + indices_num_dims; ++dim_no) {
            _idx_it_strides[dim_no] = indices_strides[dim_no - _axis];
        }

        for (; dim_no < BICoordinates::num_max_dimensions; ++dim_no) {
            _src_it_strides[dim_no] = input_strides[dim_no - indices_num_dims + 1];
        }
    }


    void BINEGatherKernel::configure(const BIITensor *input, const BIITensor *indices, BIITensor *output, int axis) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output, indices);
        BI_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), indices->info(), output->info(), axis));

        _input = input;
        _indices = indices;
        _output = output;
        _axis = axis;

        if (_axis < 0) {
            _axis += input->info()->num_dimensions();
        }
        BI_COMPUTE_ERROR_ON(0 > _axis || _axis >= static_cast<int32_t>(input->info()->num_dimensions()));

        switch (_indices->info()->data_type()) {
            case BIDataType::U32:
                _func = &BINEGatherKernel::gather_common<uint32_t>;
                break;
            case BIDataType::S32:
                _func = &BINEGatherKernel::gather_common<int32_t>;
                break;
            default:
                BI_COMPUTE_ERROR("Not supported");
                break;
        }

        // Output auto initialization if not yet initialized
        const BITensorShape output_shape = BatmanInfer::misc::shape_calculator::compute_gather_shape(
            input->info()->tensor_shape(), indices->info()->tensor_shape(), _axis);
        auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

        // Create window: (Window进行动态修改)
        const BIWindow win = calculate_max_window(*output->info(), BISteps());

        BIINEKernel::configure(win);

        // Create input and indices strides that have the same number of dimensions as the output tensor.
        // These will be used to iterate lock-step through all tensors (input, indices and output).
        size_t dim_no = 0;

        const auto input_info = input->info();
        const auto &input_strides = input_info->strides_in_bytes();

        const auto indices_info = indices->info();
        const auto &indices_strides = indices_info->strides_in_bytes();
        const auto indices_num_dims = indices_info->num_dimensions();

        for (; dim_no < static_cast<size_t>(_axis); ++dim_no) {
            _src_it_strides[dim_no] = input_strides[dim_no];
        }

        for (; dim_no < static_cast<size_t>(_axis) + indices_num_dims; ++dim_no) {
            _idx_it_strides[dim_no] = indices_strides[dim_no - _axis];
        }

        for (; dim_no < BICoordinates::num_max_dimensions; ++dim_no) {
            _src_it_strides[dim_no] = input_strides[dim_no - indices_num_dims + 1];
        }
    }

    BIStatus
    BINEGatherKernel::validate(const BIITensorInfo *input, const BIITensorInfo *indices, const BIITensorInfo *output,
                               int axis) {
        BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, indices, output, axis));
        return BIStatus{};
    }

    void BINEGatherKernel::run(const BIWindow &window, const ThreadInfo &info) {
        BI_COMPUTE_UNUSED(info);
        BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
        BI_COMPUTE_ERROR_ON(_func == nullptr);

        (this->*_func)(window, info);
    }
}
