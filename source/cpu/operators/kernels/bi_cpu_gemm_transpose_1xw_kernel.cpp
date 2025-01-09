//
// Created by Mason on 2025/1/9.
//

#include <cpu/kernels/bi_cpu_gemm_transpose_1xw_kernel.hpp>

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

            void BICpuGemmTranspose1xWKernel::configure(const BIITensorInfo *src,
                                                        BIITensorInfo *dst) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);

                auto_init_if_empty(*dst,
                                   src->clone()->set_tensor_shape(compute_transpose_1xw_with_element_size_shape(*src)));

                // 运行验证步骤
                BI_COMPUTE_ERROR_THROW_ON(BICpuGemmTranspose1xWKernel::validate(src, dst));

                const size_t vector_size = 16 / src->element_size();

                // 配置内核窗口啊
                BIWindow win = calculate_max_window(*src, BISteps(vector_size));
                BIICPPKernel::configure(win);
            }

            BIStatus BICpuGemmTranspose1xWKernel::validate(const BatmanInfer::BIITensorInfo *src,
                                                           const BatmanInfer::BIITensorInfo *dst) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
                BI_COMPUTE_RETURN_ERROR_ON(src->data_type() == BIDataType::UNKNOWN);
                //Note: BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src) is not needed here as this kernel doesn't use CPU FP16 instructions.

                if (dst->total_size() != 0) {
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(),
                                                                      compute_transpose_1xw_with_element_size_shape(
                                                                              *src));
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
                }

                return BIStatus{};
            }

            void BICpuGemmTranspose1xWKernel::run_op(BIITensorPack &tensors,
                                                     const BIWindow &window,
                                                     const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIIKernel::window(), window);
                BI_COMPUTE_ERROR_ON(tensors.empty());

                /*
                 * Following an example of how the transposition1xW works when the src data type is F32
                 *
                 *         |a00 a01 a02 a03|
                 *         |a10 a11 a12 a13|
                 *         |a20 a21 a22 a23| = | a00 a01 a02 a03 || a10 a11 a12 a13 || a20 a21 a22 a23 || a30 a31 a32 a33 |
                 *         |a30 a31 a32 a33|
                 *
                 * The dst matrix will have the following shape: [ height * W, ceil(width / W) ], where W = (16 / element size of the tensor)
                 */

                // 设置 dst 张量的窗口。将 X 和 Y 维度设置为 0，以便允许多线程实现和未来的批量矩阵乘法
                BIWindow win_out(window);
                win_out.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                win_out.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));

                const BIITensor *src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                BIITensor *dst = tensors.get_tensor(BITensorType::ACL_DST);

                BIIterator in(src, window);
                BIIterator out(dst, win_out);

                const size_t in_width = src->info()->dimension(0);
                const size_t element_size = src->info()->element_size();
                const size_t out_stride = dst->info()->strides_in_bytes()[1];
                const size_t vector_size = 16 / element_size;

                execute_window_loop(
                        window,
                        [&](const BICoordinates &id) {
                            const uint8_t *in_ptr = in.ptr();
                            uint8_t *const out_ptr =
                                    out.ptr() + (id.y() * vector_size) * element_size +
                                    (id.x() / vector_size) * out_stride;

                            for (size_t k = 0; k < vector_size; ++k) {
                                // If the src width is not multiple of W, we fill the reference with 0s
                                if ((id.x() + k) >= in_width) {
                                    std::memset(out_ptr + k * element_size, 0, element_size);
                                } else {
                                    std::memcpy(out_ptr + k * element_size, in_ptr + k * element_size, element_size);
                                }
                            }
                        },
                        in, out);
            }

            const char *BICpuGemmTranspose1xWKernel::name() const {
                return "BICpuGemmTranspose1xWKernel";
            }
        } // namespace kernels
    }
}