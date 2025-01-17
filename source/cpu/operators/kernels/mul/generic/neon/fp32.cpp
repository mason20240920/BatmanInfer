//
// Created by Mason on 2025/1/17.
//


#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>
#include <cpu/bi_cpu_types.hpp>
#include <cpu/kernels/mul/generic/neon/fp32.hpp>

namespace BatmanInfer {
    namespace cpu {
        void
        mul_F32_F32_F32(const BIITensor *src1, const BIITensor *src2, BIITensor *out, const BIWindow &window,
                        float scale) {
            // Create input windows
            BIWindow input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
            BIWindow input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

            // Clear X Dimension on execution window as we handle manually
            BIWindow win = window;
            win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            constexpr int window_step_x = 16 / sizeof(float);
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());
            const bool is_broadcast_across_x = src1->info()->tensor_shape().x() != src2->info()->tensor_shape().x();

            using ExactTagType = typename wrapper::traits::neon_vector<float, window_step_x>::tag_type;

            if (is_broadcast_across_x) {
                const bool is_broadcast_input_2 = input2_win.x().step() == 0;
                BIWindow broadcast_win = is_broadcast_input_2 ? input2_win : input1_win;
                BIWindow non_broadcast_win = !is_broadcast_input_2 ? input2_win : input1_win;
                const BIITensor *broadcast_tensor = is_broadcast_input_2 ? src2 : src1;
                const BIITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src2 : src1;

                // Clear X Dimension on execution window as we handle manually
                non_broadcast_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                BIIterator broadcast_input(broadcast_tensor, broadcast_win);
                BIIterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
                BIIterator dst(out, win);

                execute_window_loop(
                        win,
                        [&](const BICoordinates &) {
                            const auto non_broadcast_input_ptr = reinterpret_cast<const float *>(non_broadcast_input.ptr());
                            const auto output_ptr = reinterpret_cast<float *>(dst.ptr());

                            const float broadcast_value = *reinterpret_cast<const float *>(broadcast_input.ptr());
                            const auto broadcast_value_vec = wrapper::vdup_n(broadcast_value, ExactTagType{});
                            const auto scale_vec = wrapper::vdup_n(scale, ExactTagType{});

                            // Compute window_step_x elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                const auto non_broadcast_v = wrapper::vloadq(non_broadcast_input_ptr + x);
                                auto res = wrapper::vmul(wrapper::vmul(broadcast_value_vec, non_broadcast_v),
                                                         scale_vec);
                                wrapper::vstore(output_ptr + x, res);
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                const auto non_broadcast_v = *(non_broadcast_input_ptr + x);
                                *(output_ptr + x) = broadcast_value * non_broadcast_v * scale;
                            }
                        },
                        broadcast_input, non_broadcast_input, dst);
            } else {
                // Clear X Dimension on execution window as we handle manually
                input1_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                input2_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                BIIterator input1(src1, input1_win);
                BIIterator input2(src2, input2_win);
                BIIterator dst(out, win);

                execute_window_loop(
                        win,
                        [&](const BICoordinates &) {
                            const auto input1_ptr = reinterpret_cast<const float *>(input1.ptr());
                            const auto input2_ptr = reinterpret_cast<const float *>(input2.ptr());
                            const auto output_ptr = reinterpret_cast<float *>(dst.ptr());

                            // Compute window_step_x elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                const auto ta1 = wrapper::vloadq(input1_ptr + x);
                                const auto ta2 = wrapper::vloadq(input2_ptr + x);
                                const auto scale_vec = wrapper::vdup_n(scale, ExactTagType{});
                                const auto res = wrapper::vmul(wrapper::vmul(ta1, ta2), scale_vec);
                                wrapper::vstore(output_ptr + x, res);
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                const auto ta1 = *(input1_ptr + x);
                                const auto ta2 = *(input2_ptr + x);
                                *(output_ptr + x) = ta1 * ta2 * scale;
                            }
                        },
                        input1, input2, dst);
            }
        }
    } // namespace cpu
} // namespace BatmanInfer