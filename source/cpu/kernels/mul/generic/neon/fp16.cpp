//
// Created by Mason on 2025/1/17.
//

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "data/core/cpp/bi_cpp_validate.hpp"
#include "data/core/neon/wrapper/wrapper.hpp"
#include "cpu/bi_cpu_types.hpp"
#include "cpu/kernels/mul/generic/neon/fp16.hpp"

namespace BatmanInfer {
    namespace cpu {
        void mul_F16_F16_F16(const BIITensor *src1,
                             const BIITensor *src2,
                             BIITensor *out,
                             const BIWindow &window,
                             float scale) {
            // 创建输入的窗口(如果维度是小于1, 都设置为广播)
            BIWindow input1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());
            BIWindow input2_win = window.broadcast_if_dimension_le_one(src2->info()->tensor_shape());

            // 清空在执行窗口的x维度
            BIWindow win = window;
            win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
            // 窗口x维度的步数是16
            constexpr int window_step_x = 16;
            // 窗口x维度开始的点
            const auto window_start_x = static_cast<int>(window.x().start());
            // 窗口x维度结束的点
            const auto window_end_x = static_cast<int>(window.x().end());
            const bool is_broadcast_across_x = src1->info()->tensor_shape().x() != src2->info()->tensor_shape().x();
            if (is_broadcast_across_x) {
                // 是否需要广播输入2的维度x
                const bool is_broadcast_input_2 = input2_win.x().step() == 0;
                BIWindow broadcast_win = is_broadcast_input_2 ? input2_win : input1_win;
                BIWindow non_broadcast_win = !is_broadcast_input_2 ? input2_win : input1_win;
                const BIITensor *broadcast_tensor = is_broadcast_input_2 ? src2 : src1;
                const BIITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src2 : src1;
                // 清空x维度
                non_broadcast_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                BIIterator broadcast_input(broadcast_tensor, broadcast_win);
                BIIterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
                BIIterator dst(out, win);
                execute_window_loop(
                        win,
                        [&](const BICoordinates &) {
                            const auto non_broadcast_input_ptr = reinterpret_cast<const float16_t *>(non_broadcast_input.ptr());
                            const auto output_ptr = reinterpret_cast<float16_t *>(dst.ptr());
                            const auto broadcast_value = *reinterpret_cast<const float16_t *>(broadcast_input.ptr());
                            const float16x8x2_t broadcast_value_vec = {{
                                                                               vdupq_n_f16(broadcast_value),
                                                                               vdupq_n_f16(broadcast_value),
                                                                       }};
                            const auto scale_vec = vdupq_n_f16(scale);
                            // Compute window_step_x elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                const float16x8x2_t non_broadcast_v = {{
                                                                               vld1q_f16(non_broadcast_input_ptr + x),
                                                                               vld1q_f16(
                                                                                       non_broadcast_input_ptr + x + 8),
                                                                       }};
                                const float16x8x2_t result = {{
                                                                      vmulq_f16(vmulq_f16(broadcast_value_vec.val[0],
                                                                                          non_broadcast_v.val[0]),
                                                                                scale_vec),
                                                                      vmulq_f16(vmulq_f16(broadcast_value_vec.val[1],
                                                                                          non_broadcast_v.val[1]),
                                                                                scale_vec),
                                                              }};
                                vst1q_f16(output_ptr + x, result.val[0]);
                                vst1q_f16(output_ptr + x + 8, result.val[1]);
                            }
                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                const auto non_broadcast_v = *(non_broadcast_input_ptr + x);
                                *(output_ptr + x) = broadcast_value * non_broadcast_v * scale;
                            }
                        },
                        broadcast_input, non_broadcast_input, dst);
            } else {
                input1_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                input2_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                BIIterator input1(src1, input1_win);
                BIIterator input2(src2, input2_win);
                BIIterator dst(out, win);
                execute_window_loop(
                        win,
                        [&](const BICoordinates &) {
                            const auto input1_ptr = reinterpret_cast<const float16_t *>(input1.ptr());
                            const auto input2_ptr = reinterpret_cast<const float16_t *>(input2.ptr());
                            const auto output_ptr = reinterpret_cast<float16_t *>(dst.ptr());
                            // Compute window_step_x elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                const float16x8x2_t ta1 = {{
                                                                   vld1q_f16(input1_ptr + x),
                                                                   vld1q_f16(input1_ptr + x + 8),
                                                           }};
                                const float16x8x2_t ta2 = {{
                                                                   vld1q_f16(input2_ptr + x),
                                                                   vld1q_f16(input2_ptr + x + 8),
                                                           }};
                                const float16x8_t scale_vec = vdupq_n_f16(scale);
                                const float16x8x2_t result = {{
                                                                      vmulq_f16(vmulq_f16(ta1.val[0], ta2.val[0]),
                                                                                scale_vec),
                                                                      vmulq_f16(vmulq_f16(ta1.val[1], ta2.val[1]),
                                                                                scale_vec),
                                                              }};
                                vst1q_f16(output_ptr + x, result.val[0]);
                                vst1q_f16(output_ptr + x + 8, result.val[1]);
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
    }
}

#endif