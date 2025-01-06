//
// Created by Mason on 2025/1/6.
//

#include <cpu/kernels/bi_cpu_transpose_kernel.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_helpers.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                unsigned int num_elems_processed(size_t element_size) {
                    switch (element_size) {
                        case 1:
                            return 8;
                        case 2:
                            return 4;
                        case 4:
#ifdef __aarch64__
                            return 8;
#else  // __aarch64__
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
                    const int    window_step_x            = 8;
                    const int    window_step_y            = 8;
                    const int    window_start_x           = window.x().start();
                    const int    window_end_x             = window.x().end();
                    const int    window_start_y           = window.y().start();
                    const int    window_end_y             = std::min(window.y().end(),
                                                                     static_cast<int>(in->info()->dimension(1)));
                    const int    window_end_y_multiple_of =
                                         ((window_end_y - window_start_y) / window_step_y) * window_step_y;
                    const size_t input_stride_in_bytes    = in->info()->strides_in_bytes()[1];
                    const size_t output_stride_in_bytes   = out->info()->strides_in_bytes()[1];

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
                }
            }
        }
    }
}