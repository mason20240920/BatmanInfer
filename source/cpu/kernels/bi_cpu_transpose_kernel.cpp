//
// Created by Mason on 2025/1/6.
//

#include "cpu/kernels/bi_cpu_transpose_kernel.hpp"
#include "data/core/bi_i_tensor.hpp"
#include "data/core/bi_helpers.hpp"
#include "data/core/bi_vlidate.hpp"
#include "data/core/utils/misc/bi_shape_calculator.hpp"
#include "data/core/helpers/bi_auto_configuration.hpp"
#include "data/core/bi_steps.hpp"
#include "data/core/helpers/bi_window_helpers.hpp"

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                /**
                 * 用于根据输入的 元素大小（element_size）返回处理的元素数量
                 * @param element_size 表示元素的大小（单位: 字节）
                 * @return
                 */
                unsigned int num_elems_processed(size_t element_size) {
                    switch (element_size) {
                        case 1:
                            return 8;
                        case 2:
                            return 4;
                        case 4:
                            // 如果代码运行在 AArch64 架构（64 位 ARM 架构）上，则一次处理 8 个元素
#ifdef __aarch64__
                            return 8;
#else  // __aarch64__
                            // 非 __aarch64__: 如果代码运行在其他架构（如 32 位 ARM 架构）上，则一次处理 4 个元素
                            return 4;
#endif // __aarch64__
                        default:
                            break;
                    }

                    BI_COMPUTE_ERROR("Element size not supported!");
                }

                void transpose_8bit_elements(const BIITensor *in,
                                             BIITensor *out,
                                             const BIWindow &window) {
                    // 2. 核心步骤
                    // 2.1 窗口初始化
                    //   * 定义处理块的大小 (8 x 8)
                    //   * 通过window 确定了待处理区域的起始和结束坐标
                    //   * 计算了 y 维度的边界 window_end_y_multiple_of，确保可以整除 8，方便 SIMD 优化
                    const int window_step_x = 8;
                    const int window_step_y = 8;
                    const int window_start_x = window.x().start();
                    const int window_end_x = window.x().end();
                    const int window_start_y = window.y().start();
                    const int window_end_y = std::min(window.y().end(),
                                                      static_cast<int>(in->info()->dimension(1)));
                    const int window_end_y_multiple_of =
                            ((window_end_y - window_start_y) / window_step_y) * window_step_y;
                    const size_t input_stride_in_bytes = in->info()->strides_in_bytes()[1];
                    const size_t output_stride_in_bytes = out->info()->strides_in_bytes()[1];

                    // 2.2 检查是否有剩余行
                    //   * 判断 y 方向是否存在无法整除 8 的剩余行
                    bool left_over_loop_y = (((window_end_y - window_start_y) % window_step_y) != 0);

                    // 2.3 窗口配置
                    //   * window_in 确定了 x 和 y 的处理范围，用于后续的 SIMD 优化
                    //   * 如果存在剩余行，则调整 y 的范围，确保主循环只处理完整的 8x8 块
                    BIWindow window_in(window);
                    window_in.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                    if (left_over_loop_y) {
                        if (window_end_y_multiple_of > window_step_y)
                            window_in.set(BIWindow::DimY,
                                          BIWindow::BIDimension(window_start_y, window_end_y_multiple_of,
                                                                window_step_y));
                        else
                            window_in.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 1));
                    }

                    BIWindow window_out(window);
                    window_out.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                    window_out.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));

                    BIIterator output(out, window_out);

                    // 仅当输入不是行向量时，才运行SIMD路径。
                    if (in->info()->dimension(1) != 1) {
                        BIIterator input(in, window_in);
                        execute_window_loop(
                                window_in,
                                [&](const BICoordinates &id) {
                                    // 每次迭代计算 8x8 个元素
                                    // uint8x8_t 这是 ARM NEON 的 SIMD 数据类型，表示一个 8 位无符号整数的向量，包含 8 个元素
                                    // 一次可以加载 8 个字节的数据

                                    // vld1_u8
                                    // NEON 的指令，用于从内存中加载 8 个无符号 8 位整数到向量寄存器中
                                    int x = window_start_x;
                                    for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                        const uint8x8_t row0 =
                                                vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x +
                                                                                          0 * input_stride_in_bytes));
                                        const uint8x8_t row1 =
                                                vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x +
                                                                                          1 * input_stride_in_bytes));
                                        const uint8x8_t row2 =
                                                vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x +
                                                                                          2 * input_stride_in_bytes));
                                        const uint8x8_t row3 =
                                                vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x +
                                                                                          3 * input_stride_in_bytes));
                                        const uint8x8_t row4 =
                                                vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x +
                                                                                          4 * input_stride_in_bytes));
                                        const uint8x8_t row5 =
                                                vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x +
                                                                                          5 * input_stride_in_bytes));
                                        const uint8x8_t row6 =
                                                vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x +
                                                                                          6 * input_stride_in_bytes));
                                        const uint8x8_t row7 =
                                                vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + x +
                                                                                          7 * input_stride_in_bytes));

                                        // 转置 2x2
                                        // 对加载的数据进行 2x2 的交叉转置，为后续操作（如矩阵乘法）做准备
                                        // 对于 row0 和 row1：
                                        // 输入：row0 = [a0, a1, a2, a3, a4, a5, a6, a7] row1 = [b0, b1, b2, b3, b4, b5, b6, b7]
                                        // 输出：
                                        // 偶数位：k0_u8.val[0] = [a0, b0, a2, b2, a4, b4, a6, b6]
                                        // 奇数位：k0_u8.val[1] = [a1, b1, a3, b3, a5, b5, a7, b7]
                                        const uint8x8x2_t k0_u8 = vtrn_u8(row0, row1);
                                        const uint8x8x2_t k1_u8 = vtrn_u8(row2, row3);
                                        const uint8x8x2_t k2_u8 = vtrn_u8(row4, row5);
                                        const uint8x8x2_t k3_u8 = vtrn_u8(row6, row7);

                                        // Transpose 4x4
                                        const uint16x4x2_t k0_u16 =
                                                vtrn_u16(vreinterpret_u16_u8(k0_u8.val[0]),
                                                         vreinterpret_u16_u8(k1_u8.val[0]));
                                        const uint16x4x2_t k1_u16 =
                                                vtrn_u16(vreinterpret_u16_u8(k0_u8.val[1]),
                                                         vreinterpret_u16_u8(k1_u8.val[1]));
                                        const uint16x4x2_t k2_u16 =
                                                vtrn_u16(vreinterpret_u16_u8(k2_u8.val[0]),
                                                         vreinterpret_u16_u8(k3_u8.val[0]));
                                        const uint16x4x2_t k3_u16 =
                                                vtrn_u16(vreinterpret_u16_u8(k2_u8.val[1]),
                                                         vreinterpret_u16_u8(k3_u8.val[1]));

                                        // Transpose 8x8
                                        const uint32x2x2_t k0_u32 =
                                                vtrn_u32(vreinterpret_u32_u16(k0_u16.val[0]),
                                                         vreinterpret_u32_u16(k2_u16.val[0]));
                                        const uint32x2x2_t k1_u32 =
                                                vtrn_u32(vreinterpret_u32_u16(k0_u16.val[1]),
                                                         vreinterpret_u32_u16(k2_u16.val[1]));
                                        const uint32x2x2_t k2_u32 =
                                                vtrn_u32(vreinterpret_u32_u16(k1_u16.val[0]),
                                                         vreinterpret_u32_u16(k3_u16.val[0]));
                                        const uint32x2x2_t k3_u32 =
                                                vtrn_u32(vreinterpret_u32_u16(k1_u16.val[1]),
                                                         vreinterpret_u32_u16(k3_u16.val[1]));

                                        // Compute destination address
                                        const size_t dst_offset_in_bytes =
                                                id.y() * sizeof(uint8_t) + x * output_stride_in_bytes;

                                        vst1_u8(
                                                reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes +
                                                                            0 * output_stride_in_bytes),
                                                vreinterpret_u8_u16(vreinterpret_u16_u32(k0_u32.val[0])));
                                        vst1_u8(
                                                reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes +
                                                                            1 * output_stride_in_bytes),
                                                vreinterpret_u8_u16(vreinterpret_u16_u32(k2_u32.val[0])));
                                        vst1_u8(
                                                reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes +
                                                                            2 * output_stride_in_bytes),
                                                vreinterpret_u8_u16(vreinterpret_u16_u32(k1_u32.val[0])));
                                        vst1_u8(
                                                reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes +
                                                                            3 * output_stride_in_bytes),
                                                vreinterpret_u8_u16(vreinterpret_u16_u32(k3_u32.val[0])));
                                        vst1_u8(
                                                reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes +
                                                                            4 * output_stride_in_bytes),
                                                vreinterpret_u8_u16(vreinterpret_u16_u32(k0_u32.val[1])));
                                        vst1_u8(
                                                reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes +
                                                                            5 * output_stride_in_bytes),
                                                vreinterpret_u8_u16(vreinterpret_u16_u32(k2_u32.val[1])));
                                        vst1_u8(
                                                reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes +
                                                                            6 * output_stride_in_bytes),
                                                vreinterpret_u8_u16(vreinterpret_u16_u32(k1_u32.val[1])));
                                        vst1_u8(
                                                reinterpret_cast<uint8_t *>(output.ptr() + dst_offset_in_bytes +
                                                                            7 * output_stride_in_bytes),
                                                vreinterpret_u8_u16(vreinterpret_u16_u32(k3_u32.val[1])));
                                    }

                                    // Compute left-over elements along the x dimension (1x8)
                                    for (; x < window_end_x; ++x) {
                                        const uint8_t val0 = *(input.ptr() + x + 0 * input_stride_in_bytes);
                                        const uint8_t val1 = *(input.ptr() + x + 1 * input_stride_in_bytes);
                                        const uint8_t val2 = *(input.ptr() + x + 2 * input_stride_in_bytes);
                                        const uint8_t val3 = *(input.ptr() + x + 3 * input_stride_in_bytes);
                                        const uint8_t val4 = *(input.ptr() + x + 4 * input_stride_in_bytes);
                                        const uint8_t val5 = *(input.ptr() + x + 5 * input_stride_in_bytes);
                                        const uint8_t val6 = *(input.ptr() + x + 6 * input_stride_in_bytes);
                                        const uint8_t val7 = *(input.ptr() + x + 7 * input_stride_in_bytes);

                                        uint8x8_t result = vdup_n_u8(0);
                                        result = vset_lane_u8(val0, result, 0);
                                        result = vset_lane_u8(val1, result, 1);
                                        result = vset_lane_u8(val2, result, 2);
                                        result = vset_lane_u8(val3, result, 3);
                                        result = vset_lane_u8(val4, result, 4);
                                        result = vset_lane_u8(val5, result, 5);
                                        result = vset_lane_u8(val6, result, 6);
                                        result = vset_lane_u8(val7, result, 7);

                                        // Compute destination address
                                        const size_t dst_offset_in_bytes =
                                                id.y() * sizeof(uint8_t) + x * output_stride_in_bytes;

                                        vst1_u8(output.ptr() + dst_offset_in_bytes, result);
                                    }
                                }, input, output);
                    }

                    if (left_over_loop_y) {
                        window_in.set(BIWindow::DimX, BIWindow::BIDimension(window.x().start(), window.x().end(), 1));
                        window_in.set(BIWindow::DimY, BIWindow::BIDimension(window_end_y_multiple_of, window_end_y, 1));

                        BIIterator input(in, window_in);
                        BIIterator output(out, window_out);

                        // Compute left-over elements along the y dimension (1x1)
                        execute_window_loop(
                                window_in,
                                [&](const BICoordinates &id) {
                                    const uint8_t val0 = *input.ptr();

                                    // Compute destination address
                                    const size_t dst_offset_in_bytes =
                                            id.y() * sizeof(uint8_t) + id.x() * output_stride_in_bytes;

                                    *(output.ptr() + dst_offset_in_bytes) = val0;
                                },
                                input, output);
                    }
                }

                void transpose_16bit_elements(const BIITensor *in, BIITensor *out, const BIWindow &window) {
                    const int window_step_x = 4;
                    const int window_step_y = 4;
                    const int window_start_x = window.x().start();
                    const int window_end_x = window.x().end();
                    const int window_start_y = window.y().start();
                    const int window_end_y = std::min(window.y().end(), static_cast<int>(in->info()->dimension(1)));
                    const int window_end_y_multiple_of =
                            ((window_end_y - window_start_y) / window_step_y) * window_step_y;
                    const size_t input_stride_in_bytes = in->info()->strides_in_bytes()[1];
                    const size_t output_stride_in_bytes = out->info()->strides_in_bytes()[1];

                    // Check if we need a left-over loop for the y dimension
                    bool left_over_loop_y = (((window_end_y - window_start_y) % window_step_y) != 0);

                    BIWindow window_in(window);
                    window_in.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                    if (left_over_loop_y) {
                        // Check if window_end_y_multiple_of is greater than window_start_y
                        if (window_end_y_multiple_of > window_start_y) {
                            window_in.set(BIWindow::DimY,
                                          BIWindow::BIDimension(window_start_y, window_end_y_multiple_of,
                                                                window_step_y));
                        } else {
                            window_in.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 1));
                        }
                    }

                    BIWindow window_out(window);
                    window_out.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                    window_out.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));

                    BIIterator output(out, window_out);

                    // Run the SIMD path if and only if the input is not a row-vector
                    if (in->info()->dimension(1) != 1) {
                        BIIterator input(in, window_in);
                        execute_window_loop(
                                window_in,
                                [&](const BICoordinates &id) {
                                    // Compute 4x4 elements per iteration
                                    int x = window_start_x;
                                    for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                        const uint16x4_t row0 =
                                                vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() +
                                                                                            0 * input_stride_in_bytes) +
                                                         x);
                                        const uint16x4_t row1 =
                                                vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() +
                                                                                            1 * input_stride_in_bytes) +
                                                         x);
                                        const uint16x4_t row2 =
                                                vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() +
                                                                                            2 * input_stride_in_bytes) +
                                                         x);
                                        const uint16x4_t row3 =
                                                vld1_u16(reinterpret_cast<const uint16_t *>(input.ptr() +
                                                                                            3 * input_stride_in_bytes) +
                                                         x);

                                        // Transpose 2x2
                                        const uint16x4x2_t k0_u16 = vtrn_u16(row0, row1);
                                        const uint16x4x2_t k1_u16 = vtrn_u16(row2, row3);

                                        // Transpose 4x4
                                        const uint32x2x2_t k0_u32 =
                                                vtrn_u32(vreinterpret_u32_u16(k0_u16.val[0]),
                                                         vreinterpret_u32_u16(k1_u16.val[0]));
                                        const uint32x2x2_t k1_u32 =
                                                vtrn_u32(vreinterpret_u32_u16(k0_u16.val[1]),
                                                         vreinterpret_u32_u16(k1_u16.val[1]));

                                        // Compute destination address
                                        const size_t dst_offset_in_bytes =
                                                id.y() * sizeof(uint16_t) + x * output_stride_in_bytes;

                                        vst1_u16(
                                                reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             0 * output_stride_in_bytes),
                                                vreinterpret_u16_u32(k0_u32.val[0]));
                                        vst1_u16(
                                                reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             1 * output_stride_in_bytes),
                                                vreinterpret_u16_u32(k1_u32.val[0]));
                                        vst1_u16(
                                                reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             2 * output_stride_in_bytes),
                                                vreinterpret_u16_u32(k0_u32.val[1]));
                                        vst1_u16(
                                                reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             3 * output_stride_in_bytes),
                                                vreinterpret_u16_u32(k1_u32.val[1]));
                                    }

                                    // Compute left-over elements (1x4)
                                    for (; x < window_end_x; ++x) {
                                        const uint16_t val0 = *(
                                                reinterpret_cast<uint16_t *>(input.ptr() + 0 * input_stride_in_bytes) +
                                                x);
                                        const uint16_t val1 = *(
                                                reinterpret_cast<uint16_t *>(input.ptr() + 1 * input_stride_in_bytes) +
                                                x);
                                        const uint16_t val2 = *(
                                                reinterpret_cast<uint16_t *>(input.ptr() + 2 * input_stride_in_bytes) +
                                                x);
                                        const uint16_t val3 = *(
                                                reinterpret_cast<uint16_t *>(input.ptr() + 3 * input_stride_in_bytes) +
                                                x);

                                        uint16x4_t result = vdup_n_u16(0);
                                        result = vset_lane_u16(val0, result, 0);
                                        result = vset_lane_u16(val1, result, 1);
                                        result = vset_lane_u16(val2, result, 2);
                                        result = vset_lane_u16(val3, result, 3);

                                        // Compute destination address
                                        const size_t dst_offset_in_bytes =
                                                id.y() * sizeof(uint16_t) + x * output_stride_in_bytes;

                                        vst1_u16(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes),
                                                 result);
                                    }
                                },
                                input, output);
                    }

                    if (left_over_loop_y) {
                        window_in.set(BIWindow::DimX, BIWindow::BIDimension(window.x().start(), window.x().end(), 1));
                        window_in.set(BIWindow::DimY, BIWindow::BIDimension(window_end_y_multiple_of, window_end_y, 1));

                        BIIterator input(in, window_in);
                        BIIterator output(out, window_out);

                        // Compute left-over elements along the y dimension (1x1)
                        execute_window_loop(
                                window_in,
                                [&](const BICoordinates &id) {
                                    const uint16_t val0 = *(reinterpret_cast<uint16_t *>(input.ptr()));

                                    // Compute destination address
                                    const size_t dst_offset_in_bytes =
                                            id.y() * sizeof(uint16_t) + id.x() * output_stride_in_bytes;

                                    *(reinterpret_cast<uint16_t *>(output.ptr() + dst_offset_in_bytes)) = val0;
                                },
                                input, output);
                    }
                }

#ifdef __aarch64__

                inline uint32x4x2_t vld1q_u32_x2_(const uint32_t *ptr) {
                    // gcc-7 doesn't support vld1q_u32_x2 instruction
                    return {vld1q_u32(ptr), vld1q_u32(ptr + 4)};
                }

                inline void vst1q_u32_x2_(const uint32_t *ptr, const uint32x4x2_t &val) {
                    // gcc-7 doesn't support vst1q_u32_x2 instruction
                    vst1q_u32(const_cast<uint32_t *>(ptr), val.val[0]);
                    vst1q_u32(const_cast<uint32_t *>(ptr + 4), val.val[1]);
                }

                void transpose_32bit_elements(const BIITensor *in, BIITensor *out, const BIWindow &window) {
                    constexpr int window_step_x = 8;
                    constexpr int window_step_y = 8;
                    const int window_start_x = window.x().start();
                    const int window_end_x = window.x().end();
                    const int window_start_y = window.y().start();
                    const int window_end_y = std::min(window.y().end(), static_cast<int>(in->info()->dimension(1)));
                    const int window_end_y_multiple_of =
                            ((window_end_y - window_start_y) / window_step_y) * window_step_y;
                    const size_t input_stride_in_bytes = in->info()->strides_in_bytes()[1];
                    const size_t output_stride_in_bytes = out->info()->strides_in_bytes()[1];

                    // Check if we need a left-over loop for the y dimension
                    bool left_over_loop_y = (((window_end_y - window_start_y) % window_step_y) != 0);

                    BIWindow window_in(window);
                    window_in.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                    if (left_over_loop_y) {
                        // Check if window_end_y_multiple_of is greater than window_start_y
                        if (window_end_y_multiple_of > window_start_y) {
                            window_in.set(BIWindow::DimY,
                                          BIWindow::BIDimension(window_start_y, window_end_y_multiple_of,
                                                                window_step_y));
                        } else {
                            window_in.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 1));
                        }
                    }

                    BIWindow window_out(window);
                    window_out.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                    window_out.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));

                    BIIterator output(out, window_out);

                    // Run the SIMD path if and only if the input is not a row-vector
                    if (in->info()->dimension(1) != 1) {
                        BIIterator input(in, window_in);
                        execute_window_loop(
                                window_in,
                                [&](const BICoordinates &id) {
                                    // Compute 8x8 elements per iteration
                                    int x = window_start_x;
                                    for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                        // Load
                                        const uint32x4x2_t row0 =
                                                vld1q_u32_x2_(reinterpret_cast<const uint32_t *>(input.ptr() + 0 *
                                                                                                               input_stride_in_bytes) +
                                                              x);
                                        const uint32x4x2_t row1 =
                                                vld1q_u32_x2_(reinterpret_cast<const uint32_t *>(input.ptr() + 1 *
                                                                                                               input_stride_in_bytes) +
                                                              x);
                                        const uint32x4x2_t row2 =
                                                vld1q_u32_x2_(reinterpret_cast<const uint32_t *>(input.ptr() + 2 *
                                                                                                               input_stride_in_bytes) +
                                                              x);
                                        const uint32x4x2_t row3 =
                                                vld1q_u32_x2_(reinterpret_cast<const uint32_t *>(input.ptr() + 3 *
                                                                                                               input_stride_in_bytes) +
                                                              x);
                                        const uint32x4x2_t row4 =
                                                vld1q_u32_x2_(reinterpret_cast<const uint32_t *>(input.ptr() + 4 *
                                                                                                               input_stride_in_bytes) +
                                                              x);
                                        const uint32x4x2_t row5 =
                                                vld1q_u32_x2_(reinterpret_cast<const uint32_t *>(input.ptr() + 5 *
                                                                                                               input_stride_in_bytes) +
                                                              x);
                                        const uint32x4x2_t row6 =
                                                vld1q_u32_x2_(reinterpret_cast<const uint32_t *>(input.ptr() + 6 *
                                                                                                               input_stride_in_bytes) +
                                                              x);
                                        const uint32x4x2_t row7 =
                                                vld1q_u32_x2_(reinterpret_cast<const uint32_t *>(input.ptr() + 7 *
                                                                                                               input_stride_in_bytes) +
                                                              x);

                                        // Transpose 2x4
                                        const uint32x4x2_t k0_u32 = {vtrn1q_u32(row0.val[0], row1.val[0]),
                                                                     vtrn2q_u32(row0.val[0], row1.val[0])};
                                        const uint32x4x2_t k1_u32 = {vtrn1q_u32(row0.val[1], row1.val[1]),
                                                                     vtrn2q_u32(row0.val[1], row1.val[1])};
                                        const uint32x4x2_t k2_u32 = {vtrn1q_u32(row2.val[0], row3.val[0]),
                                                                     vtrn2q_u32(row2.val[0], row3.val[0])};
                                        const uint32x4x2_t k3_u32 = {vtrn1q_u32(row2.val[1], row3.val[1]),
                                                                     vtrn2q_u32(row2.val[1], row3.val[1])};
                                        const uint32x4x2_t k4_u32 = {vtrn1q_u32(row4.val[0], row5.val[0]),
                                                                     vtrn2q_u32(row4.val[0], row5.val[0])};
                                        const uint32x4x2_t k5_u32 = {vtrn1q_u32(row4.val[1], row5.val[1]),
                                                                     vtrn2q_u32(row4.val[1], row5.val[1])};
                                        const uint32x4x2_t k6_u32 = {vtrn1q_u32(row6.val[0], row7.val[0]),
                                                                     vtrn2q_u32(row6.val[0], row7.val[0])};
                                        const uint32x4x2_t k7_u32 = {vtrn1q_u32(row6.val[1], row7.val[1]),
                                                                     vtrn2q_u32(row6.val[1], row7.val[1])};

                                        // Transpose 2x2
                                        const uint64x2x2_t k0_u64 = {
                                                vtrn1q_u64(vreinterpretq_u64_u32(k0_u32.val[0]),
                                                           vreinterpretq_u64_u32(k2_u32.val[0])),
                                                vtrn2q_u64(vreinterpretq_u64_u32(k0_u32.val[0]),
                                                           vreinterpretq_u64_u32(k2_u32.val[0]))};
                                        const uint64x2x2_t k1_u64 = {
                                                vtrn1q_u64(vreinterpretq_u64_u32(k0_u32.val[1]),
                                                           vreinterpretq_u64_u32(k2_u32.val[1])),
                                                vtrn2q_u64(vreinterpretq_u64_u32(k0_u32.val[1]),
                                                           vreinterpretq_u64_u32(k2_u32.val[1]))};
                                        const uint64x2x2_t k2_u64 = {
                                                vtrn1q_u64(vreinterpretq_u64_u32(k1_u32.val[0]),
                                                           vreinterpretq_u64_u32(k3_u32.val[0])),
                                                vtrn2q_u64(vreinterpretq_u64_u32(k1_u32.val[0]),
                                                           vreinterpretq_u64_u32(k3_u32.val[0]))};
                                        const uint64x2x2_t k3_u64 = {
                                                vtrn1q_u64(vreinterpretq_u64_u32(k1_u32.val[1]),
                                                           vreinterpretq_u64_u32(k3_u32.val[1])),
                                                vtrn2q_u64(vreinterpretq_u64_u32(k1_u32.val[1]),
                                                           vreinterpretq_u64_u32(k3_u32.val[1]))};
                                        const uint64x2x2_t k4_u64 = {
                                                vtrn1q_u64(vreinterpretq_u64_u32(k4_u32.val[0]),
                                                           vreinterpretq_u64_u32(k6_u32.val[0])),
                                                vtrn2q_u64(vreinterpretq_u64_u32(k4_u32.val[0]),
                                                           vreinterpretq_u64_u32(k6_u32.val[0]))};
                                        const uint64x2x2_t k5_u64 = {
                                                vtrn1q_u64(vreinterpretq_u64_u32(k4_u32.val[1]),
                                                           vreinterpretq_u64_u32(k6_u32.val[1])),
                                                vtrn2q_u64(vreinterpretq_u64_u32(k4_u32.val[1]),
                                                           vreinterpretq_u64_u32(k6_u32.val[1]))};
                                        const uint64x2x2_t k6_u64 = {
                                                vtrn1q_u64(vreinterpretq_u64_u32(k5_u32.val[0]),
                                                           vreinterpretq_u64_u32(k7_u32.val[0])),
                                                vtrn2q_u64(vreinterpretq_u64_u32(k5_u32.val[0]),
                                                           vreinterpretq_u64_u32(k7_u32.val[0]))};
                                        const uint64x2x2_t k7_u64 = {
                                                vtrn1q_u64(vreinterpretq_u64_u32(k5_u32.val[1]),
                                                           vreinterpretq_u64_u32(k7_u32.val[1])),
                                                vtrn2q_u64(vreinterpretq_u64_u32(k5_u32.val[1]),
                                                           vreinterpretq_u64_u32(k7_u32.val[1]))};

                                        // Swap blocks
                                        const uint32x4x2_t col0 = {vreinterpretq_u32_u64(k0_u64.val[0]),
                                                                   vreinterpretq_u32_u64(k4_u64.val[0])};
                                        const uint32x4x2_t col1 = {vreinterpretq_u32_u64(k1_u64.val[0]),
                                                                   vreinterpretq_u32_u64(k5_u64.val[0])};
                                        const uint32x4x2_t col2 = {vreinterpretq_u32_u64(k0_u64.val[1]),
                                                                   vreinterpretq_u32_u64(k4_u64.val[1])};
                                        const uint32x4x2_t col3 = {vreinterpretq_u32_u64(k1_u64.val[1]),
                                                                   vreinterpretq_u32_u64(k5_u64.val[1])};
                                        const uint32x4x2_t col4 = {vreinterpretq_u32_u64(k2_u64.val[0]),
                                                                   vreinterpretq_u32_u64(k6_u64.val[0])};
                                        const uint32x4x2_t col5 = {vreinterpretq_u32_u64(k3_u64.val[0]),
                                                                   vreinterpretq_u32_u64(k7_u64.val[0])};
                                        const uint32x4x2_t col6 = {vreinterpretq_u32_u64(k2_u64.val[1]),
                                                                   vreinterpretq_u32_u64(k6_u64.val[1])};
                                        const uint32x4x2_t col7 = {vreinterpretq_u32_u64(k3_u64.val[1]),
                                                                   vreinterpretq_u32_u64(k7_u64.val[1])};

                                        // Compute destination address
                                        const size_t dst_offset_in_bytes =
                                                id.y() * sizeof(uint32_t) + x * output_stride_in_bytes;

                                        // Store
                                        vst1q_u32_x2_(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             0 * output_stride_in_bytes),
                                                col0);
                                        vst1q_u32_x2_(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             1 * output_stride_in_bytes),
                                                col1);
                                        vst1q_u32_x2_(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             2 * output_stride_in_bytes),
                                                col2);
                                        vst1q_u32_x2_(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             3 * output_stride_in_bytes),
                                                col3);
                                        vst1q_u32_x2_(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             4 * output_stride_in_bytes),
                                                col4);
                                        vst1q_u32_x2_(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             5 * output_stride_in_bytes),
                                                col5);
                                        vst1q_u32_x2_(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             6 * output_stride_in_bytes),
                                                col6);
                                        vst1q_u32_x2_(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes +
                                                                             7 * output_stride_in_bytes),
                                                col7);
                                    }

                                    // Compute left-over elements (8x1)
                                    for (; x < window_end_x; ++x) {
                                        const uint32_t val0 = *(
                                                reinterpret_cast<uint32_t *>(input.ptr() + 0 * input_stride_in_bytes) +
                                                x);
                                        const uint32_t val1 = *(
                                                reinterpret_cast<uint32_t *>(input.ptr() + 1 * input_stride_in_bytes) +
                                                x);
                                        const uint32_t val2 = *(
                                                reinterpret_cast<uint32_t *>(input.ptr() + 2 * input_stride_in_bytes) +
                                                x);
                                        const uint32_t val3 = *(
                                                reinterpret_cast<uint32_t *>(input.ptr() + 3 * input_stride_in_bytes) +
                                                x);
                                        const uint32_t val4 = *(
                                                reinterpret_cast<uint32_t *>(input.ptr() + 4 * input_stride_in_bytes) +
                                                x);
                                        const uint32_t val5 = *(
                                                reinterpret_cast<uint32_t *>(input.ptr() + 5 * input_stride_in_bytes) +
                                                x);
                                        const uint32_t val6 = *(
                                                reinterpret_cast<uint32_t *>(input.ptr() + 6 * input_stride_in_bytes) +
                                                x);
                                        const uint32_t val7 = *(
                                                reinterpret_cast<uint32_t *>(input.ptr() + 7 * input_stride_in_bytes) +
                                                x);

                                        uint32x4_t result0 = vdupq_n_u32(0);
                                        uint32x4_t result1 = vdupq_n_u32(0);
                                        result0 = vsetq_lane_u32(val0, result0, 0);
                                        result0 = vsetq_lane_u32(val1, result0, 1);
                                        result0 = vsetq_lane_u32(val2, result0, 2);
                                        result0 = vsetq_lane_u32(val3, result0, 3);
                                        result1 = vsetq_lane_u32(val4, result1, 0);
                                        result1 = vsetq_lane_u32(val5, result1, 1);
                                        result1 = vsetq_lane_u32(val6, result1, 2);
                                        result1 = vsetq_lane_u32(val7, result1, 3);

                                        // Compute destination address
                                        const size_t dst_offset_in_bytes =
                                                id.y() * sizeof(uint32_t) + x * output_stride_in_bytes;

                                        vst1q_u32_x2_(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes),
                                                      {result0, result1});
                                    }
                                },
                                input, output);
                    }

                    if (left_over_loop_y) {
                        window_in.set(BIWindow::DimX, BIWindow::BIDimension(window.x().start(), window.x().end(), 1));
                        window_in.set(BIWindow::DimY, BIWindow::BIDimension(window_end_y_multiple_of, window_end_y, 1));

                        BIIterator input(in, window_in);
                        BIIterator output(out, window_out);

                        // Compute left-over elements along the y dimension (1x1)
                        execute_window_loop(
                                window_in,
                                [&](const BICoordinates &id) {
                                    const uint32_t val0 = *(reinterpret_cast<uint32_t *>(input.ptr()));

                                    // Compute destination address
                                    const size_t dst_offset_in_bytes =
                                            id.y() * sizeof(uint32_t) + id.x() * output_stride_in_bytes;

                                    *(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes)) = val0;
                                },
                                input, output);
                    }
                }

#else
                void transpose_32bit_elements(const BIITensor *in, BIITensor *out, const BIWindow &window)
                {
                    const int window_step_x = 4;
                    const int window_step_y = 4;
                    const int window_start_x = window.x().start();
                    const int window_end_x = window.x().end();
                    const int window_start_y = window.y().start();
                    const int window_end_y = std::min(window.y().end(), static_cast<int>(in->info()->dimension(1)));
                    const int window_end_y_multiple_of = ((window_end_y - window_start_y) / window_step_y) * window_step_y;
                    const size_t input_stride_in_bytes = in->info()->strides_in_bytes()[1];
                    const size_t output_stride_in_bytes = out->info()->strides_in_bytes()[1];

                    // Check if we need a left-over loop for the y dimension
                    bool left_over_loop_y = (((window_end_y - window_start_y) % window_step_y) != 0);

                    BIWindow window_in(window);
                    window_in.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                    if (left_over_loop_y)
                    {
                        // Check if window_end_y_multiple_of is greater than window_start_y
                        if (window_end_y_multiple_of > window_start_y)
                        {
                            window_in.set(BIWindow::DimY, BIWindow::BIDimension(window_start_y, window_end_y_multiple_of, window_step_y));
                        }
                        else
                        {
                            window_in.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 1));
                        }
                    }

                    BIWindow window_out(window);
                    window_out.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                    window_out.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));

                    BIIterator output(out, window_out);

                    // Run the SIMD path if and only if the input is not a row-vector
                    if (in->info()->dimension(1) != 1)
                    {
                        BIIterator input(in, window_in);
                        execute_window_loop(
                                window_in,
                                [&](const BICoordinates &id)
                                {
                                    // Compute 4x4 elements per iteration
                                    int x = window_start_x;
                                    for (; x <= (window_end_x - window_step_x); x += window_step_x)
                                    {
                                        const uint32x4_t row0 =
                                                vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 0 * input_stride_in_bytes) + x);
                                        const uint32x4_t row1 =
                                                vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 1 * input_stride_in_bytes) + x);
                                        const uint32x4_t row2 =
                                                vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 2 * input_stride_in_bytes) + x);
                                        const uint32x4_t row3 =
                                                vld1q_u32(reinterpret_cast<const uint32_t *>(input.ptr() + 3 * input_stride_in_bytes) + x);

                                        // Transpose 2x2
                                        const uint32x2x2_t k0_u32 = vtrn_u32(vget_low_u32(row0), vget_low_u32(row1));
                                        const uint32x2x2_t k1_u32 = vtrn_u32(vget_high_u32(row2), vget_high_u32(row3));
                                        const uint32x2x2_t k2_u32 = vtrn_u32(vget_high_u32(row0), vget_high_u32(row1));
                                        const uint32x2x2_t k3_u32 = vtrn_u32(vget_low_u32(row2), vget_low_u32(row3));

                                        // Compute destination address
                                        const size_t dst_offset_in_bytes = id.y() * sizeof(uint32_t) + x * output_stride_in_bytes;

                                        // Swap block 01 with block 10 and store
                                        vst1q_u32(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 0 * output_stride_in_bytes),
                                                vcombine_u32(k0_u32.val[0], k3_u32.val[0]));
                                        vst1q_u32(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 1 * output_stride_in_bytes),
                                                vcombine_u32(k0_u32.val[1], k3_u32.val[1]));
                                        vst1q_u32(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 2 * output_stride_in_bytes),
                                                vcombine_u32(k2_u32.val[0], k1_u32.val[0]));
                                        vst1q_u32(
                                                reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes + 3 * output_stride_in_bytes),
                                                vcombine_u32(k2_u32.val[1], k1_u32.val[1]));
                                    }

                                    // Compute left-over elements (1x4)
                                    for (; x < window_end_x; ++x)
                                    {
                                        const uint32_t val0 = *(reinterpret_cast<uint32_t *>(input.ptr() + 0 * input_stride_in_bytes) + x);
                                        const uint32_t val1 = *(reinterpret_cast<uint32_t *>(input.ptr() + 1 * input_stride_in_bytes) + x);
                                        const uint32_t val2 = *(reinterpret_cast<uint32_t *>(input.ptr() + 2 * input_stride_in_bytes) + x);
                                        const uint32_t val3 = *(reinterpret_cast<uint32_t *>(input.ptr() + 3 * input_stride_in_bytes) + x);

                                        uint32x4_t result = vdupq_n_u32(0);
                                        result = vsetq_lane_u32(val0, result, 0);
                                        result = vsetq_lane_u32(val1, result, 1);
                                        result = vsetq_lane_u32(val2, result, 2);
                                        result = vsetq_lane_u32(val3, result, 3);

                                        // Compute destination address
                                        const size_t dst_offset_in_bytes = id.y() * sizeof(uint32_t) + x * output_stride_in_bytes;

                                        vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes), result);
                                    }
                                },
                                input, output);
                    }

                    if (left_over_loop_y)
                    {
                        window_in.set(BIWindow::DimX, BIWindow::BIDimension(window.x().start(), window.x().end(), 1));
                        window_in.set(BIWindow::DimY, BIWindow::BIDimension(window_end_y_multiple_of, window_end_y, 1));

                        BIIterator input(in, window_in);
                        BIIterator output(out, window_out);

                        // Compute left-over elements along the y dimension (1x1)
                        execute_window_loop(
                                window_in,
                                [&](const BICoordinates &id)
                                {
                                    const uint32_t val0 = *(reinterpret_cast<uint32_t *>(input.ptr()));

                                    // Compute destination address
                                    const size_t dst_offset_in_bytes = id.y() * sizeof(uint32_t) + id.x() * output_stride_in_bytes;

                                    *(reinterpret_cast<uint32_t *>(output.ptr() + dst_offset_in_bytes)) = val0;
                                },
                                input, output);
                    }
                }

#endif
            } // namespace

            void
            BICpuTransposeKernel::configure(const BIITensorInfo *src, BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);

                // Destination auto initialization if not yet initialized
                const BITensorShape dst_shape = misc::shape_calculator::compute_transposed_shape(*src);
                auto_init_if_empty(*dst, src->clone()->set_tensor_shape(dst_shape));

                // Explicitly set the tensor shape to preserve dimensions
                dst->set_tensor_shape(dst_shape);

                // Perform validation step
                BI_COMPUTE_ERROR_THROW_ON(validate(src, dst));

                // 注意：这个内核每次迭代处理 16 个元素。
                // 然而，由于我们在两个维度（X 和 Y）上都使用了剩余循环（left-over for loop），
                // 因此不能有任何内存的越界读取或写入。
                // 出于这个原因，num_elems_processed_per_iteration_x 被设置为 1。
                const unsigned int num_elems_processed_per_iteration_x = 1;
                const unsigned int num_elems_processed_per_iteration_y = num_elems_processed(src->element_size());

                // Configure kernel window
                BIWindow win =
                        calculate_max_window(*src, BISteps(num_elems_processed_per_iteration_x,
                                                           num_elems_processed_per_iteration_y));

                // The CpuTranspose doesn't need padding so update_window_and_padding() can be skipped
                BICoordinates coord;
                coord.set_num_dimensions(dst->num_dimensions());
                dst->set_valid_region(BIValidRegion(coord, dst->tensor_shape()));

                BIICpuKernel::configure(win);
            }

            BIStatus BICpuTransposeKernel::validate(const BatmanInfer::BIITensorInfo *src,
                                                    const BatmanInfer::BIITensorInfo *dst) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
                //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use CPU FP16 instructions.
                BI_COMPUTE_RETURN_ERROR_ON(src->data_type() == BIDataType::UNKNOWN);

                // Error if input is not 8 bit, 16bit or 32bit
                BI_COMPUTE_RETURN_ERROR_ON_MSG(
                        src->element_size() != 1 && src->element_size() != 2 && src->element_size() != 4,
                        "Element size not supported");

                // Validate configured destination
                if (dst->total_size() != 0) {
                    const BITensorShape dst_shape = misc::shape_calculator::compute_transposed_shape(*src);

                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), dst_shape);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
                }

                return BIStatus{};
            }

            void BICpuTransposeKernel::run_op(BatmanInfer::BIITensorPack &tensors, const BatmanInfer::BIWindow &window,
                                              const BatmanInfer::ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICPPKernel::window(), window);

                const auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                switch (src->info()->element_size()) {
                    case 1:
                        transpose_8bit_elements(src, dst, window);
                        break;
                    case 2:
                        transpose_16bit_elements(src, dst, window);
                        break;
                    case 4:
                        transpose_32bit_elements(src, dst, window);
                        break;
                    default:
                        BI_COMPUTE_ERROR("Element size not supported");
                        break;
                }
            }

            const char *BICpuTransposeKernel::name() const {
                return "BICpuTransposeKernel";
            }

        }
    }
}