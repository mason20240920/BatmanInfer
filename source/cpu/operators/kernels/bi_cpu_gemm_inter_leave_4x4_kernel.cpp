//
// Created by Mason on 2025/1/8.
//

#include <cpu/kernels/bi_cpu_gemm_inter_leave_4x4_kernel.hpp>

#include <data/core/bi_i_tensor.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            using namespace BatmanInfer::misc::shape_calculator;

            void BICpuGemmInterleave4x4Kernel::configure(const BIITensorInfo *src,
                                                         BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);

                // 目的张量如果没有初始化则自动初始化
                auto_init_if_empty(*dst, src->clone()->set_tensor_shape(compute_interleaved_shape(*src)));

                // 运行可用步数
                BI_COMPUTE_ERROR_THROW_ON(BICpuGemmInterleave4x4Kernel::validate(src, dst));

                BIWindow win = calculate_max_window(*src, BISteps(1, 4));
                BIICPPKernel::configure(win);
            }

            BIStatus BICpuGemmInterleave4x4Kernel::validate(const BIITensorInfo *src,
                                                            const BIITensorInfo *dst) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
                // Note: 这里不需要判断BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED, 因为不使用CPU FP16 instructions指令
                BI_COMPUTE_RETURN_ERROR_ON(src->data_type() == BIDataType::UNKNOWN);

                if (dst->total_size() != 0) {
                    const BITensorShape dst_shape = compute_interleaved_shape(*src);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), dst_shape);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
                }

                return BIStatus{};
            }

            void BICpuGemmInterleave4x4Kernel::run_op(BIITensorPack &tensors,
                                                      const BIWindow &window,
                                                      const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIIKernel::window(), window);
                BI_COMPUTE_ERROR_ON(tensors.empty());

                /**
                 * 这个内核将矩阵 A 的一个 4x4 块的值放在同一行中（交错存储值）。
                 *        |a00 a01 a02 a03|
                 *        |a10 a11 a12 a13|
                 *        |a20 a21 a22 a23| = | a00 a10 a20 a30 || a01 a11 a21 a31 || a02 a12 a22 a32 || a03 a13 a23 a33 |
                 *        |a30 a31 a32 a33|
                 *
                 * 经过此操作后，目标矩阵 (dst) 将具有以下形状： [ height * 4, ceil(width / 4.0f) ]
                 */
                const BIITensor *src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                BIITensor *dst = tensors.get_tensor(BITensorType::ACL_DST);

                const size_t window_start_x = window.x().start();
                const size_t window_end_x = window.x().end();

                const size_t in_height = src->info()->dimension(1);
                const size_t in_stride = src->info()->strides_in_bytes()[1];

                // 计算输入张量高度中未对齐到 4 的部分
                const size_t partial_y = in_height % 4;

                const size_t element_size = src->info()->element_size();

                // 给源张量设置window
                BIWindow win = window;
                win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                // 给目标张量设置Window
                BIWindow win_out(window);
                win_out.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                // 在 x 维度上设置步长为 1，在 y 维度上设置步长为 0.25 (因为交错存储会将 4 行压缩为 1 行)
                win_out.scale(BIWindow::DimY, 0.25f);

                BIIterator in(src, win);
                BIIterator out(dst, win_out);

                execute_window_loop(
                        win,
                        [&](const BICoordinates &id) {
                            if (id.y() + 4 <= static_cast<int>(in_height)) {
                                for (size_t x = window_start_x; x < window_end_x; ++x) {
                                    std::memcpy(out.ptr() + (x * 4 + 0) * element_size,
                                                (in.ptr() + 0 * in_stride) + x * element_size,
                                                element_size);
                                    std::memcpy(out.ptr() + (x * 4 + 1) * element_size,
                                                (in.ptr() + 1 * in_stride) + x * element_size,
                                                element_size);
                                    std::memcpy(out.ptr() + (x * 4 + 2) * element_size,
                                                (in.ptr() + 2 * in_stride) + x * element_size,
                                                element_size);
                                    std::memcpy(out.ptr() + (x * 4 + 3) * element_size,
                                                (in.ptr() + 3 * in_stride) + x * element_size,
                                                element_size);
                                }
                            } else {
                                // 先复制未对齐的行，再用 std::memset 将剩余的部分填充为 0
                                for (size_t x = window_start_x; x < window_end_x; ++x) {
                                    size_t y = 0;
                                    for (; y < partial_y; ++y) {
                                        std::memcpy(out.ptr() + (x * 4 + y) * element_size,
                                                    (in.ptr() + y * in_stride) + x * element_size, element_size);
                                    }
                                    for (; y < 4; ++y) {
                                        std::memset(out.ptr() + (x * 4 + y) * element_size, 0, element_size);
                                    }
                                }
                            }
                        },
                        in, out);
            }

            const char *BICpuGemmInterleave4x4Kernel::name() const {
                return "BICpuGemmInterleave4x4Kernel";
            }
        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer