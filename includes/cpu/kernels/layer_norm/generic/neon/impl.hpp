//
// Created by Mason on 2025/2/11.
//

#pragma once

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_utils.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/bi_neon_math.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>

namespace BatmanInfer {
    /**
     * LayerNorm 核心计算函数
     *
     * @param window
     * @param input
     * @param output
     * @param gamma 缩放参数 [H]
     * @param beta 偏移参数 [H]
     * @param epsilon 防止0的小数
     *
     */
    inline void layer_norm_fp16(const BIWindow &window,
                                const BIITensor *input,
                                const BIITensor *output,
                                const BIITensor *gamma,
                                const BIITensor *beta,
                                const float epsilon = 1e-5) {
        // 1. 创建窗口
        BIWindow win(window);
        win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

        BIIterator input_it(input, win);
        BIIterator output_it(output, win);

        execute_window_loop(win, [&](const BICoordinates &id) {
            auto in_ptr = reinterpret_cast<const float16_t *>(input_it.ptr());
            auto out_ptr = reinterpret_cast<float16_t *>(output_it.ptr());
            auto gamma_ptr = reinterpret_cast<const float16_t *>(gamma->buffer());
            auto beta_ptr = reinterpret_cast<const float16_t *>(beta->buffer());

            const int N = gamma->info()->dimension(0);

            float32x4_t sum_v0 = vdupq_n_f32(0.0f);
            float32x4_t sum_v1 = vdupq_n_f32(0.0f);
            int i = 0;

            for (; i <= N - 64; i += 64) {
                __builtin_prefetch(in_ptr + i + 64);

                // 加载8个向量(64个fp16元素)
                float16x8_t x0 = vld1q_f16(in_ptr + i);
                float16x8_t x1 = vld1q_f16(in_ptr + i + 8);
                float16x8_t x2 = vld1q_f16(in_ptr + i + 16);
                float16x8_t x3 = vld1q_f16(in_ptr + i + 24);
                float16x8_t x4 = vld1q_f16(in_ptr + i + 32);
                float16x8_t x5 = vld1q_f16(in_ptr + i + 40);
                float16x8_t x6 = vld1q_f16(in_ptr + i + 48);
                float16x8_t x7 = vld1q_f16(in_ptr + i + 56);

                sum_v0 = vaddq_f32(sum_v0, vcvt_f32_f16(vget_low_f16(x0)));
                sum_v0 = vaddq_f32(sum_v0, vcvt_f32_f16(vget_high_f16(x0)));
                sum_v1 = vaddq_f32(sum_v1, vcvt_f32_f16(vget_low_f16(x1)));
                sum_v1 = vaddq_f32(sum_v1, vcvt_f32_f16(vget_high_f16(x1)));

                sum_v0 = vaddq_f32(sum_v0, vcvt_f32_f16(vget_low_f16(x2)));
                sum_v0 = vaddq_f32(sum_v0, vcvt_f32_f16(vget_high_f16(x2)));
                sum_v1 = vaddq_f32(sum_v1, vcvt_f32_f16(vget_low_f16(x3)));
                sum_v1 = vaddq_f32(sum_v1, vcvt_f32_f16(vget_high_f16(x3)));

                sum_v0 = vaddq_f32(sum_v0, vcvt_f32_f16(vget_low_f16(x4)));
                sum_v0 = vaddq_f32(sum_v0, vcvt_f32_f16(vget_high_f16(x4)));
                sum_v1 = vaddq_f32(sum_v1, vcvt_f32_f16(vget_low_f16(x5)));
                sum_v1 = vaddq_f32(sum_v1, vcvt_f32_f16(vget_high_f16(x5)));

                sum_v0 = vaddq_f32(sum_v0, vcvt_f32_f16(vget_low_f16(x6)));
                sum_v0 = vaddq_f32(sum_v0, vcvt_f32_f16(vget_high_f16(x6)));
                sum_v1 = vaddq_f32(sum_v1, vcvt_f32_f16(vget_low_f16(x7)));
                sum_v1 = vaddq_f32(sum_v1, vcvt_f32_f16(vget_high_f16(x7)));
            }


            float scalar_sum = 0.0f;
            for (; i < N; i++)
                scalar_sum += (float) in_ptr[i];

            sum_v0 = vaddq_f32(sum_v0, sum_v1);
            const float32x2_t sum_half = vadd_f32(vget_low_f32(sum_v0), vget_high_f32(sum_v0));
            const float total_sum = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0) + scalar_sum;

            const float mean = total_sum / N;
            const float16_t mean_f16 = (float16_t) mean;
            const float16x8_t mean_v = vdupq_n_f16(mean_f16);

            float32x4_t sum_sq_v0 = vdupq_n_f32(0.0f);
            float32x4_t sum_sq_v1 = vdupq_n_f32(0.0f);

            i = 0;

            for (; i <= N - 64; i += 64) {
                __builtin_prefetch(in_ptr + i + 64);

                float16x8_t x0 = vld1q_f16(in_ptr + i);
                float16x8_t x1 = vld1q_f16(in_ptr + i + 8);
                float16x8_t x2 = vld1q_f16(in_ptr + i + 16);
                float16x8_t x3 = vld1q_f16(in_ptr + i + 24);
                float16x8_t x4 = vld1q_f16(in_ptr + i + 32);
                float16x8_t x5 = vld1q_f16(in_ptr + i + 40);
                float16x8_t x6 = vld1q_f16(in_ptr + i + 48);
                float16x8_t x7 = vld1q_f16(in_ptr + i + 56);

                float16x8_t diff0 = vsubq_f16(x0, mean_v);
                float16x8_t diff1 = vsubq_f16(x1, mean_v);
                float16x8_t diff2 = vsubq_f16(x2, mean_v);
                float16x8_t diff3 = vsubq_f16(x3, mean_v);
                float16x8_t diff4 = vsubq_f16(x4, mean_v);
                float16x8_t diff5 = vsubq_f16(x5, mean_v);
                float16x8_t diff6 = vsubq_f16(x6, mean_v);
                float16x8_t diff7 = vsubq_f16(x7, mean_v);

                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(diff0)), vcvt_f32_f16(vget_low_f16(diff0)));
                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(diff0)),
                                      vcvt_f32_f16(vget_high_f16(diff0)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(diff1)), vcvt_f32_f16(vget_low_f16(diff1)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(diff1)),
                                      vcvt_f32_f16(vget_high_f16(diff1)));

                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(diff2)), vcvt_f32_f16(vget_low_f16(diff2)));
                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(diff2)),
                                      vcvt_f32_f16(vget_high_f16(diff2)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(diff3)), vcvt_f32_f16(vget_low_f16(diff3)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(diff3)),
                                      vcvt_f32_f16(vget_high_f16(diff3)));

                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(diff4)), vcvt_f32_f16(vget_low_f16(diff4)));
                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(diff4)),
                                      vcvt_f32_f16(vget_high_f16(diff4)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(diff5)), vcvt_f32_f16(vget_low_f16(diff5)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(diff5)),
                                      vcvt_f32_f16(vget_high_f16(diff5)));

                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(diff6)), vcvt_f32_f16(vget_low_f16(diff6)));
                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(diff6)),
                                      vcvt_f32_f16(vget_high_f16(diff6)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(diff7)), vcvt_f32_f16(vget_low_f16(diff7)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(diff7)),
                                      vcvt_f32_f16(vget_high_f16(diff7)));
            }

            float scalar_sum_sq = 0.0f;
            for (; i < N; i++) {
                float diff = (float) in_ptr[i] - mean;
                scalar_sum_sq += diff * diff;
            }

            sum_sq_v0 = vaddq_f32(sum_sq_v0, sum_sq_v1);
            const float32x2_t sum_sq_half = vadd_f32(vget_low_f32(sum_sq_v0), vget_high_f32(sum_sq_v0));
            const float total_sum_sq = vget_lane_f32(vpadd_f32(sum_sq_half, sum_sq_half), 0) + scalar_sum_sq;

            const float variance = total_sum_sq / N;
            const float rstd = 1.0f / sqrtf(variance + epsilon);
            const float16_t rstd_f16 = (float16_t) rstd;
            const float16x8_t rstd_v = vdupq_n_f16(rstd_f16);


            i = 0;
            for (; i <= N - 32; i += 32) {
                float16x8_t x0 = vld1q_f16(in_ptr + i);
                float16x8_t g0 = vld1q_f16(gamma_ptr + i);
                float16x8_t b0 = vld1q_f16(beta_ptr + i);

                float16x8_t x1 = vld1q_f16(in_ptr + i + 8);
                float16x8_t g1 = vld1q_f16(gamma_ptr + i + 8);
                float16x8_t b1 = vld1q_f16(beta_ptr + i + 8);

                float16x8_t x2 = vld1q_f16(in_ptr + i + 16);
                float16x8_t g2 = vld1q_f16(gamma_ptr + i + 16);
                float16x8_t b2 = vld1q_f16(beta_ptr + i + 16);

                float16x8_t x3 = vld1q_f16(in_ptr + i + 24);
                float16x8_t g3 = vld1q_f16(gamma_ptr + i + 24);
                float16x8_t b3 = vld1q_f16(beta_ptr + i + 24);

                float16x8_t norm0 = vmulq_f16(vsubq_f16(x0, mean_v), rstd_v);
                float16x8_t norm1 = vmulq_f16(vsubq_f16(x1, mean_v), rstd_v);
                float16x8_t norm2 = vmulq_f16(vsubq_f16(x2, mean_v), rstd_v);
                float16x8_t norm3 = vmulq_f16(vsubq_f16(x3, mean_v), rstd_v);

                vst1q_f16(out_ptr + i, vaddq_f16(vmulq_f16(norm0, g0), b0));
                vst1q_f16(out_ptr + i + 8, vaddq_f16(vmulq_f16(norm1, g1), b1));
                vst1q_f16(out_ptr + i + 16, vaddq_f16(vmulq_f16(norm2, g2), b2));
                vst1q_f16(out_ptr + i + 24, vaddq_f16(vmulq_f16(norm3, g3), b3));
            }

            for (; i < N; i++) {
                float x = static_cast<float>(in_ptr[i]);
                float gamma_val = static_cast<float>(gamma_ptr[i]);
                float beta_val = static_cast<float>(beta_ptr[i]);

                float normalized = (x - mean) * rstd;
                float output_val = normalized * gamma_val + beta_val;

                out_ptr[i] = static_cast<float16_t>(output_val);
            }
        }, input_it, output_it);
    }
}
