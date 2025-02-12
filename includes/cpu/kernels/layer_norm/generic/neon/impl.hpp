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
     * LayerNorm 核心计算函数 (Arm Compute Library改造版)
     * @note 改造要点：
     *       - 移除局部响应相关参数，专注特征维度归一化
     *       - 集成可学习参数gamma/beta
     *       - 优化均值和方差计算策略
     * @warning 这个版本仅仅适配默认[1, sequence length, hidden size]数据格式
     *        - 数据的布局: [Hidden Size, Sequence Length]
     * @tparam T
     * @tparam S
     * @tparam axis
     * @param window
     * @param input Layout: [H, S]
     * @param output
     * @param gemma 可选的缩放参数 [H]
     * @param beta 可选的平移参数 [H]
     * @param epsilon  防止除零的小量
     */
    template<typename T, unsigned int S, bool use_beta_gamma>
    void layer_norm(const BIWindow &window,
                    const BIITensor *input,
                    BIITensor *output,
                    const BIITensor *scale,
                    const BIITensor *beta,
                    float epsilon) {
        using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

        // 配置窗口: 每个线程处理一列[单个序列位置的所有隐藏单元]
        BIWindow win = window;
        win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1)); // 隐藏维度展开
        win.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1)); // 序列维度展开

        BIWindow scale_win = BIWindow{};
        scale_win.set(BIWindow::DimX, BIWindow::BIDimension(0, scale->info()->dimension(0)));

        BIWindow beta_win = BIWindow{};
        beta_win.set(BIWindow::DimX, BIWindow::BIDimension(0, beta->info()->dimension(0)));

        BIIterator in_iter(input, win);
        BIIterator out_iter(output, win);
        BIIterator scale_iter(scale, scale_win);
        BIIterator beta_iter(beta, beta_win);

        const int hidden_size = input->info()->dimension(0);
//        const int seq_length = input->info()->dimension(1);
        const int vec_step = S;
        const int h_stride = input->info()->strides_in_bytes()[0] / sizeof(T); // 隐藏层维度步长

        execute_window_loop(win, [&](const BICoordinates &id) {
            const int seq_index = id[1]; // 当前序列位置
            const T *in_base = reinterpret_cast<const T *>(input->ptr_to_element(
                    BICoordinates(0, seq_index))); // 输入的Tensor指针
            T *out_base = reinterpret_cast<T *>(output->ptr_to_element(
                    BICoordinates(0, seq_index))); // 输出的Tensor指针

            // Phase 1: 计算统计量（跨隐藏维度）
            T sum = 0, square_sum = 0;
            int h = 0;

            // 向量化累加
            for (; h <= hidden_size - vec_step; h += vec_step) {
                const auto data = wrapper::vloadq(in_base + h * h_stride);

                // 第一步：将 float16x8_t 拆分成两个 float16x4_t
                auto low = wrapper::vgetlow(data);
                auto high = wrapper::vgethigh(data);

                // 第二步：使用成对加法逐级归约
                auto sum2 = wrapper::vpadd(low, high); // [a0+a1, a2+a3, a4+a5, a6+a7]
                auto sum1 = wrapper::vpadd(sum2, sum2); // [(a0+a1)+(a2+a3), (a4+a5)+(a6+a7), ...]
                auto sum0 = wrapper::vpadd(sum1, sum1); // [总和, 总和, ...]

                sum += wrapper::vgetlane(sum0, 0);

                // 平方和计算同理
                auto sq_low = wrapper::vmul(low, low);
                auto sq_high = wrapper::vmul(high, high);
                auto sq_sum2 = wrapper::vpadd(sq_low, sq_high);
                auto sq_sum1 = wrapper::vpadd(sq_sum2, sq_sum2);
                auto sq_sum0 = wrapper::vpadd(sq_sum1, sq_sum1);
                square_sum += wrapper::vgetlane(sq_sum0, 0);
            }

            // 标量处理尾部
            for (; h < hidden_size; ++h) {
                const T val = in_base[h * h_stride];
                sum += val;
                square_sum += val * val;
            }

            const T mean = sum / hidden_size;
            const T var = (square_sum / hidden_size) - (mean * mean);
            const T inv_std = 1.f / std::sqrt(var + epsilon);

            const auto mean_vec = wrapper::vdup_n(mean, ExactTagType{});
            const auto inv_std_vec = wrapper::vdup_n(inv_std, ExactTagType{});

            // Phase 2: 向量归一化
            h = 0;
            for (; h <= hidden_size - vec_step; h += vec_step) {
                const T *h_ptr = in_base + h * h_stride;
                T *out_ptr = out_base + h * h_stride;

                auto data = wrapper::vloadq(h_ptr);
                data = wrapper::vsub(data, mean_vec);
                data = wrapper::vmul(data, inv_std_vec);

                if (use_beta_gamma) {
                    const T *gamma_ptr = reinterpret_cast<const T *>(scale->ptr_to_element(BICoordinates(h)));
                    const T *beta_ptr = reinterpret_cast<const T *>(beta->ptr_to_element(BICoordinates(h)));
                    const auto gamma_v = wrapper::vloadq(gamma_ptr);
                    const auto beta_v = wrapper::vloadq(beta_ptr);
                    data = wrapper::vmla(beta_v, data, gamma_v);
                }

                wrapper::vstore(out_ptr, data);
            }

            // 处理尾部元素
            for (; h < hidden_size; ++h) {
                const T *h_ptr = in_base + h * h_stride;
                T *out_ptr = out_base + h * h_stride;

                T val = (*h_ptr - mean) * inv_std;
                if (use_beta_gamma) {
                    val = val * static_cast<T>(scale->ptr_to_element(BICoordinates(h))[0])
                          + static_cast<T>(beta->ptr_to_element(BICoordinates(h))[0]);
                }
                *out_ptr = val;
            }
        }, in_iter, out_iter);
    }
}