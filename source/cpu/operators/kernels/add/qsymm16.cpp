//
// Created by Mason on 2025/1/12.
//

#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_traits.hpp>

#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/wrapper/intrinsics/intrinsics.h>

namespace BatmanInfer {
    namespace cpu {
        void add_qsymm16_neon(
                const BIITensor *src0,
                const BIITensor *src1,
                BIITensor *dst,
                const BIConvertPolicy &policy,
                const BIWindow &window) {
            BI_COMPUTE_UNUSED(policy);

            // Create input windows
            BIWindow input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
            BIWindow input2_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

            // Clear X Dimension on execution window as we handle manually
            BIWindow win = window;
            win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            const int window_step_x = 8;
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());
            const bool is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();

            const BIUniformQuantizationInfo iq1_info = src0->info()->quantization_info().uniform();
            const BIUniformQuantizationInfo iq2_info = src1->info()->quantization_info().uniform();
            const BIUniformQuantizationInfo oq_info = dst->info()->quantization_info().uniform();

            const float32x4_t vscale1 = vdupq_n_f32(iq1_info.scale);
            const float32x4_t vscale2 = vdupq_n_f32(iq2_info.scale);
            const float32x4_t invvscaleo = vdupq_n_f32(1.f / oq_info.scale);

            if (is_broadcast_across_x) {
                const bool is_broadcast_input_2 = input2_win.x().step() == 0;
                BIWindow broadcast_win = is_broadcast_input_2 ? input2_win : input1_win;
                BIWindow non_broadcast_win = !is_broadcast_input_2 ? input2_win : input1_win;
                const BIITensor *broadcast_tensor = is_broadcast_input_2 ? src1 : src0;
                const BIITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src1 : src0;
                const BIUniformQuantizationInfo broadcast_qinfo = broadcast_tensor->info()->quantization_info().uniform();
                const BIUniformQuantizationInfo non_broadcast_qinfo = non_broadcast_tensor->info()->quantization_info().uniform();

                // Clear X Dimension on execution window as we handle manually
                non_broadcast_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                BIIterator broadcast_input(broadcast_tensor, broadcast_win);
                BIIterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
                BIIterator output(dst, win);

                execute_window_loop(
                        win,
                        [&](const BICoordinates &) {
                            const auto non_broadcast_input_ptr = reinterpret_cast<const int16_t *>(non_broadcast_input.ptr());
                            const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

                            const int16_t broadcast_value = *reinterpret_cast<const int16_t *>(broadcast_input.ptr());
                            const int16x8_t broadcast_value_vec = vdupq_n_s16(broadcast_value);

                            const auto bf_0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(broadcast_value_vec))),
                                                        vscale2);
                            const auto bf_1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(broadcast_value_vec))),
                                                        vscale2);
                            const float bfs = static_cast<int32_t>(broadcast_value) * broadcast_qinfo.scale;

                            // Compute S elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                const int16x8_t a = vld1q_s16(non_broadcast_input_ptr + x);
                                const auto af_0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))), vscale1);
                                const auto af_1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a))), vscale1);

                                int32x4_t rf_0{};
                                int32x4_t rf_1{};
#ifdef __aarch64__
                                rf_0 = vcvtnq_s32_f32(vmulq_f32(vaddq_f32(af_0, bf_0), invvscaleo));
                                rf_1 = vcvtnq_s32_f32(vmulq_f32(vaddq_f32(af_1, bf_1), invvscaleo));
#else  //__aarch64__
                                rf_0 = vcvtq_s32_f32(vmulq_f32(vaddq_f32(af_0, bf_0), invvscaleo));
                    rf_1 = vcvtq_s32_f32(vmulq_f32(vaddq_f32(af_1, bf_1), invvscaleo));
#endif //__aarch64__

                                const int16x8_t pa = vcombine_s16(vqmovn_s32(rf_0), vqmovn_s32(rf_1));
                                vst1q_s16(output_ptr + x, pa);
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                const float afs = static_cast<int32_t>(*(non_broadcast_input_ptr + x)) *
                                                  non_broadcast_qinfo.scale;
                                *(output_ptr + x) = quantize_qsymm16((afs + bfs), oq_info);
                            }
                        },
                        broadcast_input, non_broadcast_input, output);
            } else {
                // Clear X Dimension on execution window as we handle manually
                input1_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                input2_win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                BIIterator input1(src0, input1_win);
                BIIterator input2(src1, input2_win);
                BIIterator output(dst, win);

                execute_window_loop(
                        win,
                        [&](const BICoordinates &) {
                            const auto input1_ptr = reinterpret_cast<const int16_t *>(input1.ptr());
                            const auto input2_ptr = reinterpret_cast<const int16_t *>(input2.ptr());
                            const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

                            // Compute S elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                const int16x8_t a = vld1q_s16(input1_ptr + x);
                                const int16x8_t b = vld1q_s16(input2_ptr + x);

                                const auto af_0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))), vscale1);
                                const auto af_1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a))), vscale1);
                                const auto bf_0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(b))), vscale2);
                                const auto bf_1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(b))), vscale2);

                                int32x4_t rf_0{};
                                int32x4_t rf_1{};
#ifdef __aarch64__
                                rf_0 = vcvtnq_s32_f32(vmulq_f32(vaddq_f32(af_0, bf_0), invvscaleo));
                                rf_1 = vcvtnq_s32_f32(vmulq_f32(vaddq_f32(af_1, bf_1), invvscaleo));
#else  //__aarch64__
                                rf_0 = vcvtq_s32_f32(vmulq_f32(vaddq_f32(af_0, bf_0), invvscaleo));
                    rf_1 = vcvtq_s32_f32(vmulq_f32(vaddq_f32(af_1, bf_1), invvscaleo));
#endif //__aarch64__

                                const int16x8_t pa = vcombine_s16(vqmovn_s32(rf_0), vqmovn_s32(rf_1));
                                vst1q_s16(output_ptr + x, pa);
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                const float afs = static_cast<int32_t>((*(input1_ptr + x))) * iq1_info.scale;
                                const float bfs = static_cast<int32_t>((*(input2_ptr + x))) * iq2_info.scale;
                                *(output_ptr + x) = quantize_qsymm16((afs + bfs), dst->info()->quantization_info());
                            }
                        },
                        input1, input2, output);
            }
        }
    } // namespace cpu
}