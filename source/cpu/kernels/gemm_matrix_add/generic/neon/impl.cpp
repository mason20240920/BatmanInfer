//
// Created by Mason on 2025/1/8.
//

#include "cpu/kernels/gemm_matrix_add/generic/neon/impl.hpp"

#include "neon/neon_defines.h"

namespace BatmanInfer {
    namespace cpu {
        void matrix_addition_f32(const BIITensor *src,
                                 BIITensor *dst,
                                 const BIWindow &window,
                                 float beta) {
            BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
            const float32x4_t beta_f32 = vdupq_n_f32(beta);

            constexpr int window_step_x = 16;
            const auto window_start_x = static_cast<int>(window.x().start());
            const auto window_end_x = static_cast<int>(window.x().end());

            BIWindow win = window.collapse_if_possible(window, BIWindow::DimZ);
            win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

            BIIterator in(src, win);
            BIIterator out(dst, win);

            execute_window_loop(
                    win,
                    [&](const BICoordinates &) {
                        const auto in_ptr = reinterpret_cast<const float *>(in.ptr());
                        const auto out_ptr = reinterpret_cast<float *>(out.ptr());

                        int x = window_start_x;
                        for (; x < (window_end_x - window_step_x); x += window_step_x) {
                            float32x4x4_t alpha_ab = vld4q_f32(out_ptr + x);
                            const float32x4x4_t c = vld4q_f32(in_ptr + x);

                            // Multiply matrix C by its weight and accumulate
                            alpha_ab.val[0] = vmlaq_f32(alpha_ab.val[0], c.val[0], beta_f32);
                            alpha_ab.val[1] = vmlaq_f32(alpha_ab.val[1], c.val[1], beta_f32);
                            alpha_ab.val[2] = vmlaq_f32(alpha_ab.val[2], c.val[2], beta_f32);
                            alpha_ab.val[3] = vmlaq_f32(alpha_ab.val[3], c.val[3], beta_f32);

                            vst4q_f32(out_ptr + x, alpha_ab);
                        }

                        // Left-over loop
                        for (; x < window_end_x; ++x) {
                            *(out_ptr + x) += *(in_ptr + x) * beta;
                        }
                    },
                    in, out);
        }
    }
}