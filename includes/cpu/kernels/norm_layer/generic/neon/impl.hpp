//
// Created by Mason on 2025/2/9.
//

#pragma once

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_utils.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/helpers/bi_normalization_helpers.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/bi_neon_math.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>

namespace BatmanInfer {
    /**
     * 归一化核心计算函数（ARM Compute Library实现）
     *
     * 主要功能特点：
     * - 支持1D(X/Z轴)和2D(X-Y)局部响应归一化
     * - 使用NEON指令进行向量化加速
     * - 分块处理策略：前导串行块 + 向量化主体 + 尾部剩余块
     *
     * @tparam T 数据类型（float/half）
     * @tparam S SIMD向量宽度（如float32x4_t时S=4）
     * @tparam dim 归一化维度（0=X轴，2=Z轴）
     * @tparam do_2D_norm 是否执行二维归一化
     * @param window 并行计算窗口
     * @param in 输入张量
     * @param in_squared 输入平方张量（预计算好的输入元素平方值）
     * @param out 输出张量
     * @param n_info 归一化参数（类型/尺寸等）
     */
    template<typename T, unsigned int S, unsigned int dim, bool do_2D_norm>
    void normalize_float(const BIWindow &window,
                         const BIITensor *in,
                         const BIITensor *in_squared,
                         BIITensor *out,
                         BINormalizationLayerInfo n_info) {
        // NEON向量类型标签（用于指令派发）
        using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

        /* 窗口配置 */
        BIWindow win(window);
        win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1)); // X维度串行处理

        /* 向量化参数 */
        const auto window_start_x = static_cast<int>(window.x().start());
        const auto window_end_x = static_cast<int>(window.x().end());
        const int window_step_x = S; // SIMD向量步长

        /* 数据迭代器 */
        BIIterator input(in, win); // 输入数据迭代器
        BIIterator input_squared(in_squared, win); // 平方输入迭代器
        BIIterator output(out, win); // 输出迭代器

        /* 内存布局相关参数 */
        const int dim_y = in->info()->data_layout() == BIDataLayout::NCHW ? 1 : 2; // 根据布局确定Y维度
        const int radius = n_info.norm_size() / 2; // 归一化窗口半径

        /* 平方输入张量的步幅信息 */
        const int input_squared_stride_x = in_squared->info()->strides_in_bytes()[0];     // X轴步幅
        const int input_squared_stride_slice = in_squared->info()->strides_in_bytes()[dim];   // 切片步幅
        const int input_squared_stride_row = in_squared->info()->strides_in_bytes()[dim_y]; // 行步幅

        const int max_right = in->info()->dimension(dim) - 1;
        const int max_bottom = in->info()->dimension(dim_y) - 1;

        const auto coeff_vec = wrapper::vdup_n(static_cast<T>(n_info.scale_coeff()), ExactTagType{});
        const auto beta_vec = wrapper::vdup_n(static_cast<T>(n_info.beta()), ExactTagType{});
        const auto kappa_vec = wrapper::vdup_n(static_cast<T>(n_info.kappa()), ExactTagType{});

        // 局部归一化处理函数（Lambda表达式）
        auto sequential_normalization = [&](
                const int x, // 当前处理元素的x坐标
                const BICoordinates &id, // 多维坐标对象
                const int current_row, // 当前行号
                const int first_row, // 起始行号（处理窗口）
                const int last_row, // 结束行号（处理窗口）
                const T *input_ptr, // 输入数据指针
                const uint8_t *input_squared_start_ptr, // 平方输入数据起始指针
                T *output_ptr) { // 输出数据指针
            // 1. 确定当前切片位置和邻域范围
            const int current_slice = dim == 0 ? x : id[dim]; // 根据维度选择切片位置
            const int first_slice = std::max(current_slice - radius, 0); // 计算邻域起始切片
            const int last_slice = std::min(current_slice + radius, max_right); // 计算邻域结束切片

            // 2. 平方值累加（二维邻域内）
            const uint8_t *const input_squared_x_ptr = input_squared_start_ptr + x * input_squared_stride_x;
            // Accumulate 2D In-Map values
            // 累加器初始化
            auto accu = static_cast<T>(0.f);
            for (int j = first_row; j <= last_row; ++j) { // 行方向循环
                // Compute row displacement
                const uint8_t *const input_squared_ptr =
                        input_squared_x_ptr + (j - current_row) * input_squared_stride_row; // 切片方向循环
                for (int i = first_slice; i <= last_slice; ++i) {
                    // 类型转换后取值
                    accu +=
                            *reinterpret_cast<const T *>(input_squared_ptr +
                                                         (i - current_slice) * input_squared_stride_slice);
                }
            }

            // 3. 归一化计算与结果存储
            // 计算归一化系数
            const auto normalized =
                    std::pow(accu * static_cast<T>(n_info.scale_coeff()) + static_cast<T>(n_info.kappa()),
                             n_info.beta());
            const auto normalized_pixel = (*(input_ptr + x)) / normalized; // 归一化当前像素
            *(output_ptr + x) = normalized_pixel; // 存储结果
        };

        execute_window_loop(
                win,
                [&](const BICoordinates &id) {
                    const auto input_ptr = reinterpret_cast<const T *>(input.ptr());
                    auto output_ptr = reinterpret_cast<T *>(output.ptr());

                    // Get range to normalize
                    const int current_row = do_2D_norm ? id[dim_y] : 0;
                    const int first_row = do_2D_norm ? std::max(current_row - radius, 0) : 0;
                    const int last_row = do_2D_norm ? std::min(current_row + radius, max_bottom) : 0;

                    int x = window_start_x;
                    // Compute serially starting elements for the case x dimension is width
                    for (; x < radius && x < window_end_x && dim == 0; ++x) {
                        sequential_normalization(x, id, current_row, first_row, last_row, input_ptr,
                                                 input_squared.ptr(),
                                                 output_ptr);
                    }

                    // Compute vectorized
                    for (; x <= window_end_x - window_step_x - radius; x += window_step_x) {
                        const int current_slice = dim == 0 ? x : id[dim];
                        const int first_slice = std::max(current_slice - radius, 0);
                        const int last_slice = std::min(current_slice + radius, max_right);

                        const uint8_t *const input_squared_x_ptr = input_squared.ptr() + x * input_squared_stride_x;
                        // Accumulate 2D In-Map values
                        auto accu = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
                        for (int j = first_row; j <= last_row; ++j) {
                            // Compute row displacement
                            const uint8_t *const input_squared_ptr =
                                    input_squared_x_ptr + (j - current_row) * input_squared_stride_row;
                            for (int i = first_slice; i <= last_slice; ++i) {
                                accu = wrapper::vadd(
                                        accu, wrapper::vloadq(reinterpret_cast<const T *>(
                                                                      input_squared_ptr + (i - current_slice) *
                                                                                          input_squared_stride_slice)));
                            }
                        }

                        // Normalize
                        const auto normalized = wrapper::vpow(wrapper::vmla(kappa_vec, coeff_vec, accu), beta_vec);
                        const auto normalized_pixel = wrapper::vmul(wrapper::vloadq(input_ptr + x),
                                                                    wrapper::vinv(normalized));
                        wrapper::vstore(reinterpret_cast<T *>(output_ptr + x), normalized_pixel);
                    }

                    // Compute left-over elements
                    for (; x < window_end_x; ++x) {
                        sequential_normalization(x, id, current_row, first_row, last_row, input_ptr,
                                                 input_squared.ptr(),
                                                 output_ptr);
                    }
                },
                input, input_squared, output);
    }
} // namespace BatmanInfer