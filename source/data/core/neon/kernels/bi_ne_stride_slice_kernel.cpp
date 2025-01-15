//
// Created by Mason on 2025/1/15.
//

#include <data/core/neon/kernels/bi_ne_stride_slice_kernel.hpp>

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/helpers/bi_tensor_transform.h>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/utils/helpers/bi_bit_ops.h>

namespace BatmanInfer {
    namespace {
        BIStatus validate_arguments(const BIITensorInfo *input,
                                    const BIITensorInfo *output,
                                    const BICoordinates &starts,
                                    const BICoordinates &ends,
                                    const BiStrides &strides,
                                    int32_t begin_mask,
                                    int32_t end_mask,
                                    int32_t shrink_axis_mask) {
            BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
            BI_COMPUTE_RETURN_ERROR_ON(input->data_type() == BIDataType::UNKNOWN);

            BI_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().num_dimensions() > 4);
            BI_COMPUTE_RETURN_ERROR_ON(starts.num_dimensions() > input->num_dimensions());
            BI_COMPUTE_RETURN_ERROR_ON(ends.num_dimensions() > input->num_dimensions());
            BI_COMPUTE_RETURN_ERROR_ON(strides.num_dimensions() > input->num_dimensions());
            BI_COMPUTE_RETURN_ERROR_ON(
                    std::any_of(strides.cbegin(), strides.cbegin() + strides.num_dimensions(),
                                [](int i) { return i == 0; }));

            // Get expected output shape
            const BITensorShape exp_output_shape = BatmanInfer::misc::shape_calculator::compute_strided_slice_shape(
                    *input, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
            BI_COMPUTE_RETURN_ERROR_ON(exp_output_shape.total_size() == 0);

            // Checks output if configured
            if (output->total_size() != 0) {
                const BITensorInfo exp_output_info = output->clone()->set_tensor_shape(exp_output_shape);
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &exp_output_info);
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
            }

            return BIStatus{};
        }

        std::pair<BIStatus, BIWindow> validate_and_configure_window(const BIITensorInfo *input,
                                                                    BIITensorInfo *output,
                                                                    const BICoordinates &starts,
                                                                    const BICoordinates &ends,
                                                                    const BiStrides &strides,
                                                                    int32_t begin_mask,
                                                                    int32_t end_mask,
                                                                    int32_t shrink_axis_mask) {
            // Output tensor auto initialization if not yet initialized
            const BITensorShape output_shape = BatmanInfer::misc::shape_calculator::compute_strided_slice_shape(
                    *input, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
            auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape));

            // Create window
            BIWindow win = calculate_max_window(*output, BISteps());

            return std::make_pair(BIStatus{}, win);
        }
    } // namespace

    BINEStridedSliceKernel::BINEStridedSliceKernel() : _starts_abs(), _final_strides(), _shrink_mask() {

    }

    void BINEStridedSliceKernel::configure(const BatmanInfer::BIITensorInfo *input, BatmanInfer::BIITensorInfo *output,
                                           const BatmanInfer::BICoordinates &starts,
                                           const BatmanInfer::BICoordinates &ends,
                                           const BatmanInfer::BiStrides &strides, int32_t begin_mask, int32_t end_mask,
                                           int32_t shrink_axis_mask) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_ERROR_THROW_ON(
                validate_arguments(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask));
        _shrink_mask = shrink_axis_mask;
        const BITensorShape &input_shape = input->tensor_shape();
        BICoordinates ends_abs;
        std::tie(_starts_abs, ends_abs, _final_strides) =
                BatmanInfer::helpers::tensor_transform::calculate_strided_slice_coords(input_shape, starts, ends,
                                                                                       strides,
                                                                                       begin_mask, end_mask,
                                                                                       shrink_axis_mask);
        // Configure kernel window
        auto win_config =
                validate_and_configure_window(input, output, starts, ends, strides, begin_mask, end_mask,
                                              shrink_axis_mask);
        BI_COMPUTE_ERROR_THROW_ON(win_config.first);
        BIINEKernel::configure(win_config.second);
    }

    BIStatus BINEStridedSliceKernel::validate(const BIITensorInfo *input,
                                              const BIITensorInfo *output,
                                              const BICoordinates &starts,
                                              const BICoordinates &ends,
                                              const BiStrides &strides,
                                              int32_t begin_mask,
                                              int32_t end_mask,
                                              int32_t shrink_axis_mask) {
        BI_COMPUTE_RETURN_ON_ERROR(
                validate_arguments(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask));
        BI_COMPUTE_RETURN_ON_ERROR(
                validate_and_configure_window(input->clone().get(), output->clone().get(), starts, ends,
                                              strides, begin_mask, end_mask, shrink_axis_mask)
                        .first);

        return BIStatus{};
    }

    void BINEStridedSliceKernel::run_op(BIITensorPack &tensors,
                                        const BIWindow &window,
                                        const ThreadInfo &info) {
        BI_COMPUTE_UNUSED(info);
        BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
        BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIINEKernel::window(), window);

        const BIITensor *input = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
        const BIITensor *output = tensors.get_tensor(BITensorType::ACL_DST);

        size_t width_size = input->info()->element_size();

        const bool is_shrink_x = BatmanInfer::helpers::bit_ops::is_bit_set(_shrink_mask, 0);
        const bool is_shrink_y = BatmanInfer::helpers::bit_ops::is_bit_set(_shrink_mask, 1);
        const bool is_shrink_z = BatmanInfer::helpers::bit_ops::is_bit_set(_shrink_mask, 2);
        const bool is_shrink_w = BatmanInfer::helpers::bit_ops::is_bit_set(_shrink_mask, 3);

        unsigned int index = 0;
        const int idx_x = is_shrink_x ? 0 : index++;
        const int idx_y = is_shrink_y ? 0 : index++;
        const int idx_z = is_shrink_z ? 0 : index++;
        const int idx_w = is_shrink_w ? 0 : index;

        BiStrides shrinked_strides;
        shrinked_strides.set(0, is_shrink_x ? 0 : _final_strides[0]);
        shrinked_strides.set(1, is_shrink_y ? 0 : _final_strides[1]);
        shrinked_strides.set(2, is_shrink_z ? 0 : _final_strides[2]);
        shrinked_strides.set(3, is_shrink_w ? 0 : _final_strides[3]);

        BIWindow win = window;

        size_t length_x = win.shape()[0];

        if (_final_strides[0] == 1 && !is_shrink_x) {
            win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
            width_size = width_size * length_x;
        }

        BIIterator output_it(output, win);

        const int start_0 = _starts_abs[0];
        const int start_1 = _starts_abs[1];
        const int start_2 = _starts_abs[2];
        const int start_3 = _starts_abs[3];

        const int shrinked_stride_0 = shrinked_strides[0];
        const int shrinked_stride_1 = shrinked_strides[1];
        const int shrinked_stride_2 = shrinked_strides[2];
        const int shrinked_stride_3 = shrinked_strides[3];

        const int byte_increment_0 = static_cast<int>(input->info()->strides_in_bytes()[0]);
        const int byte_increment_1 = static_cast<int>(input->info()->strides_in_bytes()[1]);
        const int byte_increment_2 = static_cast<int>(input->info()->strides_in_bytes()[2]);
        const int byte_increment_3 = static_cast<int>(input->info()->strides_in_bytes()[3]);

        uint8_t *input_base = input->ptr_to_element(BICoordinates(0, 0, 0, 0));
        uint8_t *cur_ptr;

        execute_window_loop(
                win,
                [&](const BICoordinates &id) {
                    cur_ptr = input_base;
                    cur_ptr += (start_0 + (id[idx_x] * shrinked_stride_0)) * byte_increment_0;
                    cur_ptr += (start_1 + (id[idx_y] * shrinked_stride_1)) * byte_increment_1;
                    cur_ptr += (start_2 + (id[idx_z] * shrinked_stride_2)) * byte_increment_2;
                    cur_ptr += (start_3 + (id[idx_w] * shrinked_stride_3)) * byte_increment_3;

                    std::copy_n(cur_ptr, width_size, output_it.ptr());
                },
                output_it);
    }

}