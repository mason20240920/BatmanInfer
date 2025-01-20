//
// Created by Mason on 2025/1/9.
//

#include "data/core/bi_i_tensor.hpp"
#include "data/core/bi_window.hpp"
#include "cpu/kernels/gemm_matrix_add/generic/neon/fp16.hpp"
#include "data/core/bi_helpers.hpp"

#include "neon/neon_defines.h"

namespace BatmanInfer {
    namespace cpu {
        namespace {
            void matrix_addition_f16(const BIITensor *src,
                                     BIITensor *dst,
                                     const BIWindow &window,
                                     float beta) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                // 使用 NEON 指令集创建一个包含 8 个 16 位浮点数的向量 (float16x8_t)
                // 并将向量的所有元素初始化为 beta 的值。
                const float16x8_t beta_f16 = vdupq_n_f16(beta);

                // 定义窗口的步长为 16（每次处理 16 个元素）
                constexpr int window_step_x = 16;
                // 获取窗口的起始和结束位置（x 维度）
                const auto window_start_x = static_cast<int>(window.x().start());
                const auto window_end_x = static_cast<int>(window.x().end());

                // 尝试折叠窗口的高维度（z 维度及以上），以简化操作。
                BIWindow win = window.collapse_if_possible(window, BIWindow::DimZ);

                // 将窗口的 x 维度设置为单步操作（从 0 到 1，步长为 1）
                win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                // 创建输入和输出矩阵的迭代器，用于遍历窗口中的元素
                BIIterator in(src, win);
                BIIterator out(dst, win);

                // 使用窗口循环执行矩阵加法操作
                execute_window_loop(
                        win, // 遍历窗口的每个点
                        [&](const BICoordinates &) {
                            // 获取当前窗口中输入和输出矩阵的指针
                            const auto in_ptr = reinterpret_cast<const float16_t *>(in.ptr());
                            const auto out_ptr = reinterpret_cast<float16_t *>(out.ptr());

                            // 使用NEON向量化操作处理窗口中的数据
                            int x = window_start_x;
                            for (; x < (window_end_x - window_step_x); x += window_step_x) {
                                // 从输出矩阵中加载16个元素（分为两组8个元素）
                                float16x8x2_t alpha_ab = vld2q_f16(out_ptr + x);

                                // 从输入矩阵中加载16个元素（分为两组8个元素)
                                const float16x8x2_t c = vld2q_f16(in_ptr + x);

                                // 对输入矩阵的每个元素乘以 beta，并加到输出矩阵的对应元素上。
                                alpha_ab.val[0] = vaddq_f16(alpha_ab.val[0], vmulq_f16(c.val[0], beta_f16));
                                alpha_ab.val[1] = vaddq_f16(alpha_ab.val[1], vmulq_f16(c.val[1], beta_f16));

                                // 将计算结果存回输出矩阵
                                vst2q_f16(out_ptr + x, alpha_ab);
                            }

                            // 处理剩余的元素（不足 16 个的部分）
                            for (; x < window_end_x; ++x)
                                *(out_ptr + x) += *(in_ptr + x) * static_cast<float16_t>(beta);

                        },
                        in, out);
            }
        }

        void neon_fp16_gemm_matrix_add(const BIITensor *src,
                                       BIITensor *dst,
                                       const BIWindow &window,
                                       float beta) {
            return matrix_addition_f16(src, dst, window, beta);
        }
    }
}