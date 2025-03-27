//
// Created by Mason on 2025/3/27.
//

#include <cpu/kernels/dynamic_gemm/generic/neon/fp32.hpp>

#if defined(__aarch64__) && defined(ENABLE_FP32_KERNELS)

#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"

#endif // __aarch64__ && ENABLE_FP32_KERNEL

namespace BatmanInfer {
    namespace cpu {
#if defined(__aarch64__) && defined(ENABLE_FP32_KERNELS)

        void neon_fp32_dynamic_gemm_pack_rhs(const BIITensor *rhs,
                                             const BIITensor *bias,
                                             BIITensor *pack_b) {
            const size_t num_groups = 1;
            const size_t n = rhs->info()->tensor_shape().x();
            const size_t k = rhs->info()->tensor_shape().y();
            const size_t nr = kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
            const size_t kr = kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
            const size_t sr = kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
            const size_t rhs_stride = rhs->info()->strides_in_bytes().y(); // y轴有多少个元素
            const void *const rhs_ptr = rhs->buffer() + rhs->info()->offset_first_element_in_bytes(); // 右矩阵的首个指针位置
            const void *const bias_ptr = bias->buffer() + bias->info()->offset_first_element_in_bytes(); // 偏置值矩阵的首个指针位置
            const void *const scale = nullptr;
            void *const rhs_packed = pack_b->buffer();
            const size_t extra_bytes = 0;
            const void *const params = nullptr;
            kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(num_groups,
                                                             n, k, nr,
                                                             kr, sr, rhs_stride,
                                                             rhs_ptr, bias_ptr, scale,
                                                             rhs_packed, extra_bytes, params);
        }

        /**
         * @brief 执行NEON优化的动态浮点矩阵乘法运算 (GEMM)
         * 该函数实现了使用ARM NEON指令集优化的浮点矩阵乘法操作。函数通过窗口机制支持大矩阵的分块计算，
         * 当前实现仅支持在Y维度（行方向）上的工作负载分割。
         * @param a 输入左矩阵张量
         * @param b 输入右矩阵张量（当前未使用）
         * @param c 输入偏置张量（当前未使用）
         * @param d 输出结果张量
         * @param pack_b 预打包的右矩阵数据
         * @param window 计算窗口，定义了需要处理的矩阵区域
         */
        void neon_fp32_dynamic_gemm_run(
                const BIITensor *a, const BIITensor *b, const BIITensor *c, BIITensor *d, BIITensor *pack_b,
                const BIWindow &window) {
            BI_COMPUTE_UNUSED(b);
            BI_COMPUTE_UNUSED(c);

            // 获取矩阵维度
            // M: 输出矩阵行数
            // N: 输入矩阵的行数
            // K: 内积维度（左矩阵的列数/右矩阵的行数）
            const size_t M = d->info()->tensor_shape().y();
            const size_t N = d->info()->tensor_shape().x();
            const size_t K = a->info()->tensor_shape().x();

            // 获取输入和输出数据缓冲区的起始地址
            const uint8_t *const lhs_buf = a->buffer() + a->info()->offset_first_element_in_bytes();
            uint8_t *const dst_buf = d->buffer() + d->info()->offset_first_element_in_bytes();

            // 获取计算窗口的范围
            const size_t m_start = window.y().start();
            const size_t m_end = window.y().end();
            const size_t n_start = window.x().start();

            // As the workload is split in Y dimensions only, each window should start
            // from the beginning of a row.
            BI_COMPUTE_ASSERT(n_start == 0);

            // The window can be bigger than the size of the matrix.
            const size_t m_len_window = m_end - m_start;
            const size_t m_remainder = M - m_start;
            const size_t m_len = std::min(m_len_window, m_remainder);

            // As the workload is split in Y dimensions, LHS is processed in full rows.
            const size_t n_len = N;
            const size_t k_len = K;

            const size_t lhs_stride = a->info()->strides_in_bytes().y();
            const uint8_t *const lhs =
                    lhs_buf +
                    kai_get_lhs_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(m_start, lhs_stride);

            const size_t dst_stride_row = d->info()->strides_in_bytes().y();
            const size_t dst_stride_col = d->info()->strides_in_bytes().x();
            uint8_t *const dst = dst_buf + kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(
                    m_start, n_start, dst_stride_row);

            const uint8_t *const rhs_packed = pack_b->buffer();

            const float clamp_min = -std::numeric_limits<float>::max();
            const float clamp_max = std::numeric_limits<float>::max();

            kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(m_len, n_len, k_len, lhs, lhs_stride, rhs_packed,
                                                                       dst,
                                                                       dst_stride_row, dst_stride_col, clamp_min,
                                                                       clamp_max);
        }

        size_t neon_fp32_dynamic_gemm_size_of_packed_rhs(size_t rows, size_t columns) {
            // The 0.5.0 documentation is wrong. In a kxn matrix, k=rows and n=columns.
            return kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(columns, rows);
        }

        BIWindow neon_fp32_dynamic_gemm_window(const BIITensorInfo *dst) {
            const size_t m_step = kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
            const size_t n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();

            const BISteps steps(n_step, m_step);

            const BIWindow window = calculate_max_window(*dst, steps);

            return window;
        }

#endif
    }
}