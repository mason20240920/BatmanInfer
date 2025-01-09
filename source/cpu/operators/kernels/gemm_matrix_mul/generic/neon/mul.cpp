//
// Created by Mason on 2025/1/9.
//

#include <cpu/kernels/gemm_matrix_mul/generic/neon/mul.hpp>

#include <data/core/utils/helpers/float_ops.hpp>

#include <neon/neon_defines.h>


namespace BatmanInfer {
    namespace cpu {
        void vector_matrix_multiply_f32(
                const BIITensor *lhs,
                const BIITensor *rhs,
                BIITensor *dst,
                const BIWindow &window,
                const ThreadInfo &info,
                float alpha) {
            // 获取矩阵 B 的宽度（列数）和步长，以及向量 A 的元素数量
            const auto width_matrix_b = static_cast<int>(dst->info()->dimension(0));
            // 矩阵 B 的列数（结果向量 C 的长度）
            const auto in_b_stride =
                    static_cast<int>(rhs->info()->strides_in_bytes()[1] /
                                     data_size_from_type(rhs->info()->data_type()));
            // 矩阵 B 的列步长
            const auto num_elems_vec_a = static_cast<int>(lhs->info()->dimension(0));

            // 该实现每迭代计算 16 个元素(计算线程分块的起始点、步长和结束点)
            // 当前线程处理的起始列
            const int window_start_x = 16 * info.thread_id;
            // 每个线程间隔的列数（分块并行）
            const int window_step_x = 16 * info.num_threads;
            // 当前线程的结束列
            const int window_end_x = ceil_to_multiples(width_matrix_b - window_start_x, window_step_x) + window_start_x;

            // 初始化窗口对象
            BIWindow win_out(window);
            win_out.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
            win_out.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));

            BIWindow win_a(window);
            win_a.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
            win_a.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));

            BIWindow win_b;
            if (rhs->info()->num_dimensions() >= 3)
                win_b = window;
            win_b.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
            win_b.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));

            // 创建迭代器，用于遍历张量的元素
            BIIterator ina(lhs, win_a); // 遍历向量 A
            BIIterator inb(rhs, win_b); // 遍历矩阵 B
            BIIterator out(dst, win_out); // 遍历输出向量 C

            // 检查是否需要乘以 alpha
            const bool multiply_alpha = !(helpers::float_ops::is_one(alpha)); // 如果 alpha 不是 1，就需要额外乘以 alpha
            const float32x4_t alpha_f32 = vdupq_n_f32(alpha); // 将 alpha 转为 SIMD 格式

            // 主循环：遍历窗口
            execute_window_loop(
                    win_out,
                    [&](const BICoordinates &) {
                        int x = window_start_x;

                        // 主循环：处理 16 个元素的块
                        for (; x < (window_end_x - window_step_x); x += window_step_x) {
                            if (x >= width_matrix_b) {
                                return; // 如果超出矩阵 B 的列数，退出循环
                            }

                            // 初始化累积寄存器（用于存储点积结果）
                            float32x4_t acc0 = vdupq_n_f32(0.f); // 存储第 1 列的点积结果
                            float32x4_t acc1 = vdupq_n_f32(0.f); // 存储第 2 列的点积结果
                            float32x4_t acc2 = vdupq_n_f32(0.f); // 存储第 3 列的点积结果
                            float32x4_t acc3 = vdupq_n_f32(0.f); // 存储第 4 列的点积结果

                            // 获取向量 A 和矩阵 B 的指针
                            auto vec_a = reinterpret_cast<const float *>(ina.ptr()); // 向量 A 的起始地址
                            auto matrix_b = reinterpret_cast<const float *>(inb.ptr()) + x; // 矩阵 B 的起始地址（当前列块）

                            // 预取数据以优化内存访问（ARM 平台）
#if __arm__
                            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(vec_a))); // 预取向量 A 的数据
                asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b))); // 预取矩阵 B 的数据
                asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + in_b_stride))); // 预取矩阵 B 的下一行
#endif /* __arm__ */

                            // 处理向量 A 的主循环（每次处理 4 个元素）
                            auto vec_a_end_addr = vec_a + num_elems_vec_a; // 向量 A 的结束地址
                            for (; vec_a <= (vec_a_end_addr - 4);) // 每次处理 4 个元素
                            {
                                float32x2_t a0l = vld1_f32(vec_a); // 加载向量 A 的 2 个元素

                                // 加载矩阵 B 的块
                                float32x4_t b00 = vld1q_f32(matrix_b + 0 + 0 * in_b_stride);
                                float32x4_t b01 = vld1q_f32(matrix_b + 4 + 0 * in_b_stride);
                                float32x4_t b02 = vld1q_f32(matrix_b + 8 + 0 * in_b_stride);
                                float32x4_t b03 = vld1q_f32(matrix_b + 12 + 0 * in_b_stride);

                                float32x4_t b10 = vld1q_f32(matrix_b + 0 + 1 * in_b_stride);
                                float32x4_t b11 = vld1q_f32(matrix_b + 4 + 1 * in_b_stride);
                                float32x4_t b12 = vld1q_f32(matrix_b + 8 + 1 * in_b_stride);
                                float32x4_t b13 = vld1q_f32(matrix_b + 12 + 1 * in_b_stride);

                                // 更新累积寄存器（向量 A 的元素点积）
                                acc0 = vmlaq_lane_f32(acc0, b00, a0l, 0); // acc0 += b00 * a0
                                acc1 = vmlaq_lane_f32(acc1, b01, a0l, 0);
                                acc2 = vmlaq_lane_f32(acc2, b02, a0l, 0);
                                acc3 = vmlaq_lane_f32(acc3, b03, a0l, 0);

                                acc0 = vmlaq_lane_f32(acc0, b10, a0l, 1); // acc0 += b10 * a1
                                acc1 = vmlaq_lane_f32(acc1, b11, a0l, 1);
                                acc2 = vmlaq_lane_f32(acc2, b12, a0l, 1);
                                acc3 = vmlaq_lane_f32(acc3, b13, a0l, 1);

                                vec_a += 2; // 更新向量 A 的指针
                                matrix_b += 2 * in_b_stride; // 更新矩阵 B 的指针
                            }

                            // 剩余元素的处理（如果向量 A 的长度不是 4 的倍数）
                            for (; vec_a < vec_a_end_addr; ++vec_a) {
                                const float a0 = *vec_a;

                                const float32x4_t b00 = vld1q_f32(matrix_b + 0 + 0 * in_b_stride);
                                const float32x4_t b01 = vld1q_f32(matrix_b + 4 + 0 * in_b_stride);
                                const float32x4_t b02 = vld1q_f32(matrix_b + 8 + 0 * in_b_stride);
                                const float32x4_t b03 = vld1q_f32(matrix_b + 12 + 0 * in_b_stride);

                                acc0 = vmlaq_n_f32(acc0, b00, a0);
                                acc1 = vmlaq_n_f32(acc1, b01, a0);
                                acc2 = vmlaq_n_f32(acc2, b02, a0);
                                acc3 = vmlaq_n_f32(acc3, b03, a0);

                                matrix_b += in_b_stride;
                            }

                            // 如果需要，乘以 alpha
                            if (multiply_alpha) {
                                acc0 = vmulq_f32(acc0, alpha_f32);
                                acc1 = vmulq_f32(acc1, alpha_f32);
                                acc2 = vmulq_f32(acc2, alpha_f32);
                                acc3 = vmulq_f32(acc3, alpha_f32);
                            }

                            // 将结果存储到输出张量
                            const auto vec_out = reinterpret_cast<float *>(out.ptr()) + x;

                            vst1q_f32(vec_out + 0, acc0); // 存储第 1 列的结果
                            vst1q_f32(vec_out + 4, acc1); // 存储第 2 列的结果
                            vst1q_f32(vec_out + 8, acc2); // 存储第 3 列的结果
                            vst1q_f32(vec_out + 12, acc3); // 存储第 4 列的结果
                        }

                        // 剩余的列处理（如果列数不是 16 的倍数）
                        for (; x < window_end_x; ++x) {
                            if (x >= width_matrix_b) {
                                return;
                            }

                            float acc = 0.0f;

                            auto vec_a = reinterpret_cast<const float *>(ina.ptr());
                            auto matrix_b = reinterpret_cast<const float *>(inb.ptr()) + x;

                            auto vec_a_end_addr = vec_a + num_elems_vec_a;
                            for (; vec_a < vec_a_end_addr; ++vec_a) {
                                const float a0 = *vec_a;
                                const float b00 = *matrix_b;

                                acc += b00 * a0; // 点积计算

                                matrix_b += in_b_stride;
                            }

                            if (multiply_alpha) {
                                acc *= alpha; // 乘以 alpha
                            }

                            const auto vec_out = reinterpret_cast<float *>(out.ptr()) + x;

                            *vec_out = acc; // 存储结果
                        }
                    }, ina, inb, out);
        }

        void matrix_matrix_multiply_f32(
                const BIITensor *lhs, const BIITensor *rhs, BIITensor *dst, const BIWindow &window,
                const ThreadInfo &info,
                float alpha) {
            BI_COMPUTE_UNUSED(info);
            const int out_width = static_cast<int>(dst->info()->dimension(0));
            const int out_height = static_cast<int>(dst->info()->dimension(1));
            const size_t in_b_stride =
                    rhs->info()->strides_in_bytes()[1] / data_size_from_type(rhs->info()->data_type());
            const size_t out_stride1 =
                    dst->info()->strides_in_bytes()[1] / data_size_from_type(dst->info()->data_type());
            const size_t out_stride2 = out_stride1 * 2;
            const size_t out_stride3 = out_stride1 * 3;
            const int num_elems_matrix_b_x = rhs->info()->dimension(0);

            // Set step_x and step_y for matrix A. Scale by a factor of 4 the Y range as the input interleaved matrix A has 4 times less the rows of the dst matrix
            BIWindow win_a(window);
            win_a.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
            win_a.set(BIWindow::DimY,
                      BIWindow::BIDimension(window.y().start() / 4, std::max(window.y().end() / 4, 1), 1));

            BIWindow win_b;
            // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
            // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
            if (rhs->info()->num_dimensions() >= 3) {
                win_b = window;
            }
            // Set step_x and step_y for matrix B. Scale by a factor of 4 the X range as the input transposed matrix A has 4 times less the cols of the dst matrix
            // The step along the x direction is 2 times the in_b_stride because for each iteration we compute 2 blocks of size 4x4
            win_b.set(BIWindow::DimX,
                      BIWindow::BIDimension(window.x().start() / 4, window.x().end() / 4, 2 * in_b_stride));
            win_b.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));

            BIIterator ina(lhs, win_a);
            BIIterator inb(rhs, win_b);
            BIIterator out(dst, window);

            // End address of matrix B at batch number n
            const float *end_addr_mtx_b_at_batch_n =
                    reinterpret_cast<const float *>(inb.ptr()) + rhs->info()->dimension(0) * rhs->info()->dimension(1);
            std::vector<const float *> end_addr_mtx_b_per_batch = {};
            const bool multiply_alpha = !(helpers::float_ops::is_one(alpha));
            const float32x4_t alpha_f32 = vdupq_n_f32(alpha);
            const size_t out_dim2 = static_cast<int>(dst->info()->dimension(2));

            for (size_t b = 0; b < out_dim2; ++b) {
                // Store the ptrs to the last elem in the tensor for each batch
                end_addr_mtx_b_per_batch.push_back(end_addr_mtx_b_at_batch_n);
                end_addr_mtx_b_at_batch_n +=
                        rhs->info()->dimension(2) != 1 ? rhs->info()->dimension(0) * rhs->info()->dimension(1) : 0;
            }

            // The implementation assumes that the matrix A and Matrix B have been reshaped respectively with CpuGemmInterleave4x4 and CpuGemmTranspose1xW
            // The reshaping of the matrices helps to have a cache friendly implementation and helps to avoid the data re-arrangements needed for computing 16x4 elements per iteration
            // All the values needed for computing a single 4x4 block will be read from consecutive memory positions
            execute_window_loop(
                    window,
                    [&](const BICoordinates &id) {
                        auto mtx_a0 = reinterpret_cast<const float *>(ina.ptr());
                        auto mtx_b0 = reinterpret_cast<const float *>(inb.ptr());
                        auto mtx_b1 = mtx_b0 + in_b_stride;

                        float32x4_t acc00 = vdupq_n_f32(0.f);
                        float32x4_t acc10 = vdupq_n_f32(0.f);
                        float32x4_t acc20 = vdupq_n_f32(0.f);
                        float32x4_t acc30 = vdupq_n_f32(0.f);

                        float32x4_t acc01 = vdupq_n_f32(0.f);
                        float32x4_t acc11 = vdupq_n_f32(0.f);
                        float32x4_t acc21 = vdupq_n_f32(0.f);
                        float32x4_t acc31 = vdupq_n_f32(0.f);

#if __arm__
                        asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
            asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
            asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b1)));
#endif /* __arm__ */

                        auto mtx_b0_end_addr = mtx_b0 + num_elems_matrix_b_x;

                        BI_COMPUTE_ERROR_ON(end_addr_mtx_b_per_batch.size() == 0);
                        if (mtx_b1 < end_addr_mtx_b_per_batch[id.z()]) {
                            for (; mtx_b0 < (mtx_b0_end_addr - 32);) {
                                float32x4_t a0 = vld1q_dup_f32(mtx_a0 + 0);
                                float32x4_t a1 = vld1q_dup_f32(mtx_a0 + 1);
                                float32x4_t a2 = vld1q_dup_f32(mtx_a0 + 2);
                                float32x4_t a3 = vld1q_dup_f32(mtx_a0 + 3);

                                float32x4_t b00 = vld1q_f32(mtx_b0);
                                float32x4_t b10 = vld1q_f32(mtx_b1);
                                float32x4_t b01 = vld1q_f32(mtx_b0 + 4);
                                float32x4_t b11 = vld1q_f32(mtx_b1 + 4);

#if __arm__
                                asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
                    asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
                    asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b1)));
#endif /* __arm__ */

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b00, a0);
                                acc10 = vmlaq_f32(acc10, b00, a1);
                                acc20 = vmlaq_f32(acc20, b00, a2);
                                acc30 = vmlaq_f32(acc30, b00, a3);

                                float32x4_t a4 = vld1q_dup_f32(mtx_a0 + 4);
                                float32x4_t a5 = vld1q_dup_f32(mtx_a0 + 5);
                                float32x4_t a6 = vld1q_dup_f32(mtx_a0 + 6);
                                float32x4_t a7 = vld1q_dup_f32(mtx_a0 + 7);

                                // 4x4 block 1
                                acc01 = vmlaq_f32(acc01, b10, a0);
                                acc11 = vmlaq_f32(acc11, b10, a1);
                                acc21 = vmlaq_f32(acc21, b10, a2);
                                acc31 = vmlaq_f32(acc31, b10, a3);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b01, a4);
                                acc10 = vmlaq_f32(acc10, b01, a5);
                                acc20 = vmlaq_f32(acc20, b01, a6);
                                acc30 = vmlaq_f32(acc30, b01, a7);

                                // 4x4 block 1
                                acc01 = vmlaq_f32(acc01, b11, a4);
                                acc11 = vmlaq_f32(acc11, b11, a5);
                                acc21 = vmlaq_f32(acc21, b11, a6);
                                acc31 = vmlaq_f32(acc31, b11, a7);

                                mtx_a0 += 8;
                                mtx_b0 += 8;
                                mtx_b1 += 8;

                                a0 = vld1q_dup_f32(mtx_a0 + 0);
                                a1 = vld1q_dup_f32(mtx_a0 + 1);
                                a2 = vld1q_dup_f32(mtx_a0 + 2);
                                a3 = vld1q_dup_f32(mtx_a0 + 3);

                                b00 = vld1q_f32(mtx_b0);
                                b10 = vld1q_f32(mtx_b1);
                                b01 = vld1q_f32(mtx_b0 + 4);
                                b11 = vld1q_f32(mtx_b1 + 4);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b00, a0);
                                acc10 = vmlaq_f32(acc10, b00, a1);
                                acc20 = vmlaq_f32(acc20, b00, a2);
                                acc30 = vmlaq_f32(acc30, b00, a3);

                                a4 = vld1q_dup_f32(mtx_a0 + 4);
                                a5 = vld1q_dup_f32(mtx_a0 + 5);
                                a6 = vld1q_dup_f32(mtx_a0 + 6);
                                a7 = vld1q_dup_f32(mtx_a0 + 7);

                                // 4x4 block 1
                                acc01 = vmlaq_f32(acc01, b10, a0);
                                acc11 = vmlaq_f32(acc11, b10, a1);
                                acc21 = vmlaq_f32(acc21, b10, a2);
                                acc31 = vmlaq_f32(acc31, b10, a3);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b01, a4);
                                acc10 = vmlaq_f32(acc10, b01, a5);
                                acc20 = vmlaq_f32(acc20, b01, a6);
                                acc30 = vmlaq_f32(acc30, b01, a7);

                                // 4x4 block 1
                                acc01 = vmlaq_f32(acc01, b11, a4);
                                acc11 = vmlaq_f32(acc11, b11, a5);
                                acc21 = vmlaq_f32(acc21, b11, a6);
                                acc31 = vmlaq_f32(acc31, b11, a7);

                                mtx_a0 += 8;
                                mtx_b0 += 8;
                                mtx_b1 += 8;

                                a0 = vld1q_dup_f32(mtx_a0 + 0);
                                a1 = vld1q_dup_f32(mtx_a0 + 1);
                                a2 = vld1q_dup_f32(mtx_a0 + 2);
                                a3 = vld1q_dup_f32(mtx_a0 + 3);
                                b00 = vld1q_f32(mtx_b0);
                                b10 = vld1q_f32(mtx_b1);
                                b01 = vld1q_f32(mtx_b0 + 4);
                                b11 = vld1q_f32(mtx_b1 + 4);

#if __arm__
                                asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
                    asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
                    asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b1)));
#endif /* __arm__ */

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b00, a0);
                                acc10 = vmlaq_f32(acc10, b00, a1);
                                acc20 = vmlaq_f32(acc20, b00, a2);
                                acc30 = vmlaq_f32(acc30, b00, a3);

                                a4 = vld1q_dup_f32(mtx_a0 + 4);
                                a5 = vld1q_dup_f32(mtx_a0 + 5);
                                a6 = vld1q_dup_f32(mtx_a0 + 6);
                                a7 = vld1q_dup_f32(mtx_a0 + 7);

                                // 4x4 block 1
                                acc01 = vmlaq_f32(acc01, b10, a0);
                                acc11 = vmlaq_f32(acc11, b10, a1);
                                acc21 = vmlaq_f32(acc21, b10, a2);
                                acc31 = vmlaq_f32(acc31, b10, a3);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b01, a4);
                                acc10 = vmlaq_f32(acc10, b01, a5);
                                acc20 = vmlaq_f32(acc20, b01, a6);
                                acc30 = vmlaq_f32(acc30, b01, a7);

                                // 4x4 block 1
                                acc01 = vmlaq_f32(acc01, b11, a4);
                                acc11 = vmlaq_f32(acc11, b11, a5);
                                acc21 = vmlaq_f32(acc21, b11, a6);
                                acc31 = vmlaq_f32(acc31, b11, a7);

                                mtx_a0 += 8;
                                mtx_b0 += 8;
                                mtx_b1 += 8;

                                a0 = vld1q_dup_f32(mtx_a0 + 0);
                                a1 = vld1q_dup_f32(mtx_a0 + 1);
                                a2 = vld1q_dup_f32(mtx_a0 + 2);
                                a3 = vld1q_dup_f32(mtx_a0 + 3);
                                b00 = vld1q_f32(mtx_b0);
                                b10 = vld1q_f32(mtx_b1);
                                b01 = vld1q_f32(mtx_b0 + 4);
                                b11 = vld1q_f32(mtx_b1 + 4);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b00, a0);
                                acc10 = vmlaq_f32(acc10, b00, a1);
                                acc20 = vmlaq_f32(acc20, b00, a2);
                                acc30 = vmlaq_f32(acc30, b00, a3);

                                a4 = vld1q_dup_f32(mtx_a0 + 4);
                                a5 = vld1q_dup_f32(mtx_a0 + 5);
                                a6 = vld1q_dup_f32(mtx_a0 + 6);
                                a7 = vld1q_dup_f32(mtx_a0 + 7);

                                // 4x4 block 1
                                acc01 = vmlaq_f32(acc01, b10, a0);
                                acc11 = vmlaq_f32(acc11, b10, a1);
                                acc21 = vmlaq_f32(acc21, b10, a2);
                                acc31 = vmlaq_f32(acc31, b10, a3);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b01, a4);
                                acc10 = vmlaq_f32(acc10, b01, a5);
                                acc20 = vmlaq_f32(acc20, b01, a6);
                                acc30 = vmlaq_f32(acc30, b01, a7);

                                // 4x4 block 1
                                acc01 = vmlaq_f32(acc01, b11, a4);
                                acc11 = vmlaq_f32(acc11, b11, a5);
                                acc21 = vmlaq_f32(acc21, b11, a6);
                                acc31 = vmlaq_f32(acc31, b11, a7);

                                mtx_a0 += 8;
                                mtx_b0 += 8;
                                mtx_b1 += 8;
                            }

                            // Only consider one row from matrix b if subsequent row is out of boundary.
                            for (; mtx_b0 < mtx_b0_end_addr;) {
                                float32x4_t a0 = vld1q_dup_f32(mtx_a0 + 0);
                                float32x4_t a1 = vld1q_dup_f32(mtx_a0 + 1);
                                float32x4_t a2 = vld1q_dup_f32(mtx_a0 + 2);
                                float32x4_t a3 = vld1q_dup_f32(mtx_a0 + 3);
                                float32x4_t b00 = vld1q_f32(mtx_b0);
                                float32x4_t b10 = vld1q_f32(mtx_b1);

#if __arm__
                                asm volatile("PLD [%0, #128*2]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
                    asm volatile("PLD [%0, #128*2]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
                    asm volatile("PLD [%0, #128*2]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b1)));
#endif /* __arm__ */
                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b00, a0);
                                acc10 = vmlaq_f32(acc10, b00, a1);
                                acc20 = vmlaq_f32(acc20, b00, a2);
                                acc30 = vmlaq_f32(acc30, b00, a3);

                                // 4x4 block 1
                                acc01 = vmlaq_f32(acc01, b10, a0);
                                acc11 = vmlaq_f32(acc11, b10, a1);
                                acc21 = vmlaq_f32(acc21, b10, a2);
                                acc31 = vmlaq_f32(acc31, b10, a3);

                                mtx_a0 += 4;
                                mtx_b0 += 4;
                                mtx_b1 += 4;
                            }
                        }

                            // Leftover last row in matrix b, in case of there are odd number of rows in matrix B
                        else if (mtx_b0 < end_addr_mtx_b_per_batch[id.z()]) {
                            for (; mtx_b0 < (mtx_b0_end_addr - 32);) {
                                float32x4_t a0 = vld1q_dup_f32(mtx_a0 + 0);
                                float32x4_t a1 = vld1q_dup_f32(mtx_a0 + 1);
                                float32x4_t a2 = vld1q_dup_f32(mtx_a0 + 2);
                                float32x4_t a3 = vld1q_dup_f32(mtx_a0 + 3);

                                float32x4_t b00 = vld1q_f32(mtx_b0);
                                float32x4_t b01 = vld1q_f32(mtx_b0 + 4);

#if __arm__
                                asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
                    asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
#endif /* __arm__ */

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b00, a0);
                                acc10 = vmlaq_f32(acc10, b00, a1);
                                acc20 = vmlaq_f32(acc20, b00, a2);
                                acc30 = vmlaq_f32(acc30, b00, a3);

                                float32x4_t a4 = vld1q_dup_f32(mtx_a0 + 4);
                                float32x4_t a5 = vld1q_dup_f32(mtx_a0 + 5);
                                float32x4_t a6 = vld1q_dup_f32(mtx_a0 + 6);
                                float32x4_t a7 = vld1q_dup_f32(mtx_a0 + 7);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b01, a4);
                                acc10 = vmlaq_f32(acc10, b01, a5);
                                acc20 = vmlaq_f32(acc20, b01, a6);
                                acc30 = vmlaq_f32(acc30, b01, a7);

                                mtx_a0 += 8;
                                mtx_b0 += 8;

                                a0 = vld1q_dup_f32(mtx_a0 + 0);
                                a1 = vld1q_dup_f32(mtx_a0 + 1);
                                a2 = vld1q_dup_f32(mtx_a0 + 2);
                                a3 = vld1q_dup_f32(mtx_a0 + 3);

                                b00 = vld1q_f32(mtx_b0);
                                b01 = vld1q_f32(mtx_b0 + 4);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b00, a0);
                                acc10 = vmlaq_f32(acc10, b00, a1);
                                acc20 = vmlaq_f32(acc20, b00, a2);
                                acc30 = vmlaq_f32(acc30, b00, a3);

                                a4 = vld1q_dup_f32(mtx_a0 + 4);
                                a5 = vld1q_dup_f32(mtx_a0 + 5);
                                a6 = vld1q_dup_f32(mtx_a0 + 6);
                                a7 = vld1q_dup_f32(mtx_a0 + 7);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b01, a4);
                                acc10 = vmlaq_f32(acc10, b01, a5);
                                acc20 = vmlaq_f32(acc20, b01, a6);
                                acc30 = vmlaq_f32(acc30, b01, a7);

                                mtx_a0 += 8;
                                mtx_b0 += 8;

                                a0 = vld1q_dup_f32(mtx_a0 + 0);
                                a1 = vld1q_dup_f32(mtx_a0 + 1);
                                a2 = vld1q_dup_f32(mtx_a0 + 2);
                                a3 = vld1q_dup_f32(mtx_a0 + 3);
                                b00 = vld1q_f32(mtx_b0);
                                b01 = vld1q_f32(mtx_b0 + 4);

#if __arm__
                                asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
                    asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
#endif /* __arm__ */

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b00, a0);
                                acc10 = vmlaq_f32(acc10, b00, a1);
                                acc20 = vmlaq_f32(acc20, b00, a2);
                                acc30 = vmlaq_f32(acc30, b00, a3);

                                a4 = vld1q_dup_f32(mtx_a0 + 4);
                                a5 = vld1q_dup_f32(mtx_a0 + 5);
                                a6 = vld1q_dup_f32(mtx_a0 + 6);
                                a7 = vld1q_dup_f32(mtx_a0 + 7);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b01, a4);
                                acc10 = vmlaq_f32(acc10, b01, a5);
                                acc20 = vmlaq_f32(acc20, b01, a6);
                                acc30 = vmlaq_f32(acc30, b01, a7);

                                mtx_a0 += 8;
                                mtx_b0 += 8;

                                a0 = vld1q_dup_f32(mtx_a0 + 0);
                                a1 = vld1q_dup_f32(mtx_a0 + 1);
                                a2 = vld1q_dup_f32(mtx_a0 + 2);
                                a3 = vld1q_dup_f32(mtx_a0 + 3);
                                b00 = vld1q_f32(mtx_b0);
                                b01 = vld1q_f32(mtx_b0 + 4);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b00, a0);
                                acc10 = vmlaq_f32(acc10, b00, a1);
                                acc20 = vmlaq_f32(acc20, b00, a2);
                                acc30 = vmlaq_f32(acc30, b00, a3);

                                a4 = vld1q_dup_f32(mtx_a0 + 4);
                                a5 = vld1q_dup_f32(mtx_a0 + 5);
                                a6 = vld1q_dup_f32(mtx_a0 + 6);
                                a7 = vld1q_dup_f32(mtx_a0 + 7);

                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b01, a4);
                                acc10 = vmlaq_f32(acc10, b01, a5);
                                acc20 = vmlaq_f32(acc20, b01, a6);
                                acc30 = vmlaq_f32(acc30, b01, a7);

                                mtx_a0 += 8;
                                mtx_b0 += 8;
                            }
                            for (; mtx_b0 < mtx_b0_end_addr;) {
                                float32x4_t a0 = vld1q_dup_f32(mtx_a0 + 0);
                                float32x4_t a1 = vld1q_dup_f32(mtx_a0 + 1);
                                float32x4_t a2 = vld1q_dup_f32(mtx_a0 + 2);
                                float32x4_t a3 = vld1q_dup_f32(mtx_a0 + 3);
                                float32x4_t b00 = vld1q_f32(mtx_b0);

#if __arm__
                                asm volatile("PLD [%0, #128*2]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_a0)));
                    asm volatile("PLD [%0, #128*2]" ::"r"(reinterpret_cast<const uint8_t *>(mtx_b0)));
#endif /* __arm__ */
                                // 4x4 block 0
                                acc00 = vmlaq_f32(acc00, b00, a0);
                                acc10 = vmlaq_f32(acc10, b00, a1);
                                acc20 = vmlaq_f32(acc20, b00, a2);
                                acc30 = vmlaq_f32(acc30, b00, a3);

                                mtx_a0 += 4;
                                mtx_b0 += 4;
                            }
                        }

                        // Multiply by the weight of matrix product (alpha)
                        if (multiply_alpha) {
                            acc00 = vmulq_f32(acc00, alpha_f32);
                            acc10 = vmulq_f32(acc10, alpha_f32);
                            acc20 = vmulq_f32(acc20, alpha_f32);
                            acc30 = vmulq_f32(acc30, alpha_f32);
                            acc01 = vmulq_f32(acc01, alpha_f32);
                            acc11 = vmulq_f32(acc11, alpha_f32);
                            acc21 = vmulq_f32(acc21, alpha_f32);
                            acc31 = vmulq_f32(acc31, alpha_f32);
                        }

                        const auto mtx_out0 = reinterpret_cast<float *>(out.ptr());
                        const auto mtx_out1 = mtx_out0 + 4;

                        if (id.x() < (out_width - 8)) {
                            vst1q_f32(mtx_out0, acc00);
                            vst1q_f32(mtx_out1, acc01);
                            if (id.y() + 1 < out_height) {
                                vst1q_f32(mtx_out0 + out_stride1, acc10);
                                vst1q_f32(mtx_out1 + out_stride1, acc11);
                                if (id.y() + 2 < out_height) {
                                    vst1q_f32(mtx_out0 + out_stride2, acc20);
                                    vst1q_f32(mtx_out1 + out_stride2, acc21);
                                    if (id.y() + 3 < out_height) {
                                        vst1q_f32(mtx_out0 + out_stride3, acc30);
                                        vst1q_f32(mtx_out1 + out_stride3, acc31);
                                    }
                                }
                            }
                        } else if (id.x() < (out_width - 4)) {
                            vst1q_f32(mtx_out0, acc00);
                            if (id.y() + 1 < out_height) {
                                vst1q_f32(mtx_out0 + out_stride1, acc10);
                                if (id.y() + 2 < out_height) {
                                    vst1q_f32(mtx_out0 + out_stride2, acc20);
                                    if (id.y() + 3 < out_height) {
                                        vst1q_f32(mtx_out0 + out_stride3, acc30);
                                    }
                                }
                            }
                            // Left-over columns
                            const int columns_left = out_width - id.x() - 4;
                            for (auto x = 0; x < columns_left; ++x) {
                                *(mtx_out1 + x) = acc01[x];
                                if (id.y() + 1 < out_height) {
                                    *(mtx_out1 + x + out_stride1) = acc11[x];
                                    if (id.y() + 2 < out_height) {
                                        *(mtx_out1 + x + out_stride2) = acc21[x];
                                        if (id.y() + 3 < out_height) {
                                            *(mtx_out1 + x + out_stride3) = acc31[x];
                                        }
                                    }
                                }
                            }
                        } else {
                            // Left-over columns
                            const int columns_left = out_width - id.x();
                            for (int x = 0; x < columns_left; ++x) {
                                *(mtx_out0 + x) = acc00[x];
                                if (id.y() + 1 < out_height) {
                                    *(mtx_out0 + x + out_stride1) = acc10[x];
                                    if (id.y() + 2 < out_height) {
                                        *(mtx_out0 + x + out_stride2) = acc20[x];
                                        if (id.y() + 3 < out_height) {
                                            *(mtx_out0 + x + out_stride3) = acc30[x];
                                        }
                                    }
                                }
                            }
                        }
                    },
                    ina, inb, out);
        }
    }
}