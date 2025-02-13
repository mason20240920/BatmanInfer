//
// Created by Mason on 2025/2/12.
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
    * RMSNorm 核心计算函数 (Arm Compute Library改造版)
    * @note 改造要点：
    *       - 移除局部响应相关参数，专注特征维度归一化
    *       - 集成可学习参数gamma
    *       - 优化均值和方差计算策略
    * @warning 这个版本仅仅适配默认[1, sequence length, hidden size]数据格式
    *        - 数据的布局: [Hidden Size, Sequence Length]
    * @param window
    * @param input Layout: [H, S]
    * @param output
    * @param gemma 可选的缩放参数 [H]
    * @param epsilon  防止除零的小量
    */
    void rms_norm_fp16(const BIWindow &window,
                       const BIITensor *input,
                       BIITensor *output,
                       const BIITensor *gamma,
                       float epsilon = 1e-5) {
        // 创建窗口
        BIWindow win(window);
        win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

        BIIterator input_it(input, win);
        BIIterator output_it(output, win);

        execute_window_loop(win, [&](const BICoordinates &id) {
            // 获取输入/输出的数据指针
            auto in_ptr = reinterpret_cast<const float16_t *>(input_it.ptr());
            auto out_ptr = reinterpret_cast<float16_t *>(output_it.ptr());

            // 阶段1: 平方和计算 (优化指令级并行) --------------------------------

            // 双累加器初始化（每个窗口迭代独立）
            float32x4_t sum_sq_v0 = vdupq_n_f32(0.0f);
            float32x4_t sum_sq_v1 = vdupq_n_f32(0.0f);
            int i = 0;
            const int N = input->info()->dimension(0);

            // 循环展开4次 (32 elements/iteration), 减少循环开销
            for (; i <= N - 64; i += 64) {
                // 预取下个缓存行（ARM典型缓存行64字节）
                __builtin_prefetch(input + i + 64);

                // 加载8个向量（64字节）
                float16x8_t x0 = vld1q_f16(in_ptr + i);
                float16x8_t x1 = vld1q_f16(in_ptr + i + 8);
                float16x8_t x2 = vld1q_f16(in_ptr + i + 16);
                float16x8_t x3 = vld1q_f16(in_ptr + i + 24);
                float16x8_t x4 = vld1q_f16(in_ptr + i + 32);
                float16x8_t x5 = vld1q_f16(in_ptr + i + 40);
                float16x8_t x6 = vld1q_f16(in_ptr + i + 48);
                float16x8_t x7 = vld1q_f16(in_ptr + i + 56);

                // 交错计算以隐藏指令延迟
                // 转换为 float32，并计算平方累加
                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(x0)), vcvt_f32_f16(vget_low_f16(x0)));
                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(x0)), vcvt_f32_f16(vget_high_f16(x0)));

                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(x1)), vcvt_f32_f16(vget_low_f16(x1)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(x1)), vcvt_f32_f16(vget_high_f16(x1)));

                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(x2)), vcvt_f32_f16(vget_low_f16(x2)));
                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(x2)), vcvt_f32_f16(vget_high_f16(x2)));

                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(x3)), vcvt_f32_f16(vget_low_f16(x3)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(x3)), vcvt_f32_f16(vget_high_f16(x3)));

                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(x4)), vcvt_f32_f16(vget_low_f16(x4)));
                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(x4)), vcvt_f32_f16(vget_high_f16(x4)));

                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(x5)), vcvt_f32_f16(vget_low_f16(x5)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(x5)), vcvt_f32_f16(vget_high_f16(x5)));

                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(x6)), vcvt_f32_f16(vget_low_f16(x6)));
                sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(x6)), vcvt_f32_f16(vget_high_f16(x6)));

                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(x7)), vcvt_f32_f16(vget_low_f16(x7)));
                sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(x7)), vcvt_f32_f16(vget_high_f16(x7)));
            }

            // 合并累加器
            sum_sq_v0 = vaddq_f32(sum_sq_v0, sum_sq_v1);

            // 阶段2: 归约计算 (保持高精度) -------------------------------------
            float32x2_t sum_half = vadd_f32(vget_low_f32(sum_sq_v0), vget_high_f32(sum_sq_v0));
            float sum_sq = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);


            // 阶段3: 计算缩放因子 ---------------------------------------------
            const auto eps_f32 = (float) epsilon; // 正确使用标量转换函数
            const float rms_inv = 1.0f / sqrtf(sum_sq / N + eps_f32);

            const float16_t rms_inv_f16 = vduph_lane_f16(vcvt_f16_f32(vdupq_n_f32(rms_inv)), 0);
            const float16x8_t rms_inv_v = vdupq_n_f16(rms_inv_f16); // 向量化的rms_inv



            auto gamma_ptr = reinterpret_cast<const float16_t *>(gamma->buffer());
            i = 0;
            for (; i <= N - 32; i += 32) {
                // 加载input和gamma的4个向量
                float16x8_t x0 = vld1q_f16(in_ptr + i);
                float16x8_t g0 = vld1q_f16(gamma_ptr + i);
                float16x8_t x1 = vld1q_f16(in_ptr + i + 8);
                float16x8_t g1 = vld1q_f16(gamma_ptr + i + 8);
                float16x8_t x2 = vld1q_f16(in_ptr + i + 16);
                float16x8_t g2 = vld1q_f16(gamma_ptr + i + 16);
                float16x8_t x3 = vld1q_f16(in_ptr + i + 24);
                float16x8_t g3 = vld1q_f16(gamma_ptr + i + 24);

                // 计算：output = (x * gamma) * rms_inv
                vst1q_f16(out_ptr + i, vmulq_f16(vmulq_f16(x0, g0), rms_inv_v));
                vst1q_f16(out_ptr + i + 8, vmulq_f16(vmulq_f16(x1, g1), rms_inv_v));
                vst1q_f16(out_ptr + i + 16, vmulq_f16(vmulq_f16(x2, g2), rms_inv_v));
                vst1q_f16(out_ptr + i + 24, vmulq_f16(vmulq_f16(x3, g3), rms_inv_v));
            }

        }, input_it, output_it);
    }
}