//
// Created by Mason on 2025/2/6.
//

#include <cpu/kernels/gemm_matrix_mul/generic/neon/s8.hpp>
#include <neon/neon_defines.h>
#include "data/core/bi_helpers.hpp"

namespace BatmanInfer {
    namespace cpu {
        namespace {
            void vector_matrix_multiply_s8(
                    const BIITensor *lhs,
                    const BIITensor *rhs,
                    BIITensor *dst,
                    const BIWindow &window,
                    const ThreadInfo &info) {
                // 获取矩阵 B 的宽度 (输出向量的长度)
                const auto width_matrix_b = static_cast<int>(dst->info()->dimension(0));
                // 获取矩阵 B 每行的步数 (单位: 元素个数)
                const auto in_b_stride = static_cast<int>(rhs->info()->strides_in_bytes()[1] /
                                                          rhs->info()->element_size());
                // 获取向量 A 的元素数量
                const auto num_elems_vec_a = static_cast<int>(lhs->info()->dimension(0));

                // 每次迭代计算32个元素
                const int window_start_x = 32 * info.thread_id; // 当前线程负责的起始位置
                const int window_step_x = 32 * info.num_threads; // 每次跳过的步长
                // 计算当前线程的结束位置, 确保分块处理不会越界
                const int window_end_x =
                        ceil_to_multiples(width_matrix_b - window_step_x, window_step_x) + window_start_x;

                // 确保 (window_end_x - window_start_x) 是 window_step_x 的整数倍
                BI_COMPUTE_ERROR_ON_MSG((window_end_x - window_start_x) % window_step_x,
                                        " (window_end_x - window_start_x) must be multiple of window_step_x");

                // 定义输出张量的窗口，限制其在 X 和 Y 维度上的范围
                BIWindow win_out(window);
                win_out.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                win_out.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));

                // 定义向量 A 的窗口（不需要切片）
                BIWindow win_a(window);
                win_a.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                win_a.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));

                // 定义矩阵 B 的窗口
                BIWindow win_b;
                // 如果矩阵 B 的维度大于等于 3, 则保留其完整的 Z 维度
                if (rhs->info()->num_dimensions() >= 3)
                    win_b = window;
                win_b.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                win_b.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));

                // 定义三个迭代器, 分别用于向量 A, 矩阵 B和输出张量
                BIIterator ina(lhs, win_a);
                BIIterator inb(rhs, win_b);
                BIIterator out(dst, win_out);

                // 遍历输出窗口
                execute_window_loop(win_out, [&](const BICoordinates &) {
                    int x = window_start_x;
                    // 遍历当前线程负责的 X 块
                    for (; x < (window_end_x - window_step_x); x += window_step_x) {
                        if (x >= width_matrix_b)
                            return; // 如果越界，直接返回

                        // 获取矩阵 B 当前块的起始地址
                        auto matrix_b = reinterpret_cast<const int8_t *>(inb.ptr()) + x;

                        // 初始化累积变量，每个累积器使用 int32_t 表示
                        int32x4_t acc0 = vdupq_n_s32(0);
                        int32x4_t acc1 = vdupq_n_s32(0);

                        // 获取向量 A 的起始地址
                        auto vec_a = reinterpret_cast<const int8_t *>(ina.ptr());
                        const int8_t *vec_a_end_addr = vec_a + num_elems_vec_a;

                        // 内循环：每次处理 16 个元素
                        for (; vec_a <= (vec_a_end_addr - 16); vec_a += 16) {
                            // 加载向量 A 的 16 个元素
                            const int8x16_t a_vals = vld1q_s8(vec_a);

                            // 加载矩阵 B 的数据（分 2 个子块，每块 16 个元素）
                            const int8x16_t b0 = vld1q_s8(matrix_b + 0 * in_b_stride);
                            const int8x16_t b1 = vld1q_s8(matrix_b + 1 * in_b_stride);

                            // 执行乘加操作（点乘累加）
                            acc0 = vmlal_s8(acc0, vget_low_s8(a_vals), vget_low_s8(b0));
                            acc1 = vmlal_s8(acc1, vget_low_s8(a_vals), vget_low_s8(b1));

                            // 更新矩阵 B 的地址
                            matrix_b += 16;
                        }

                        // 尾部处理：不足 16 个元素的部分
                        for (; vec_a < vec_a_end_addr; ++vec_a, ++matrix_b) {
                            // 加载单个元素
                            int8_t a_val = *vec_a;
                            int8_t b_val = *matrix_b;

                            // 转换为 INT32 并累加
                            acc0 = vaddq_s32(acc0,
                                             vdupq_n_s32(static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val)));
                        }

                        // 将累积结果量化回 INT8 并存储
                        auto vec_out = reinterpret_cast<int8_t *>(out.ptr()) + x;
                        vst1_s8(vec_out + 0, vqmovn_s16(vcombine_s16(vmovn_s32(acc0), vmovn_s32(acc1))));
                    }
                }, ina, inb, out);
            }

            void matrix_matrix_multiply_s8(
                    const BIITensor *lhs,
                    const BIITensor *rhs,
                    BIITensor *dst,
                    const BIWindow &window,
                    const ThreadInfo &info) {
                BI_COMPUTE_ERROR("Run the undefined projects");
            }
        }

        void neon_s8_gemm_matrix_mul(const BIITensor *lhs,
                                     const BIITensor *rhs,
                                     BIITensor *dst,
                                     const BIWindow &window,
                                     const ThreadInfo &info,
                                     float alpha,
                                     const bool is_dst_vector) {
            return (is_dst_vector) ? vector_matrix_multiply_s8(lhs, rhs, dst, window, info)
                                   : matrix_matrix_multiply_s8(lhs, rhs, dst, window, info);
        }
    }
}