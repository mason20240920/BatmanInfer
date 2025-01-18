//
// Created by Mason on 2025/1/18.
//

#include <cpu/kernels/bi_cpu_permute_kernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_utils.hpp>

#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>

namespace {

#include <cpu/kernels/convolution/common/shims.hpp>

} // namespace

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                inline bool is_permutation_supported(const PermutationVector &v) {
                    static const std::array<PermutationVector, 2> permutations2 = {{
                                                                                           PermutationVector(0U, 1U),
                                                                                           PermutationVector(1U, 0U),
                                                                                   }};
                    static const std::array<PermutationVector, 6> permutations3 = {{
                                                                                           PermutationVector(2U, 0U,
                                                                                                             1U),
                                                                                           PermutationVector(1U, 2U,
                                                                                                             0U),
                                                                                           PermutationVector(0U, 1U,
                                                                                                             2U),
                                                                                           PermutationVector(0U, 2U,
                                                                                                             1U),
                                                                                           PermutationVector(1U, 0U,
                                                                                                             2U),
                                                                                           PermutationVector(2U, 1U,
                                                                                                             0U),
                                                                                   }};
                    static const std::array<PermutationVector, 24> permutations4 = {
                            {PermutationVector(0U, 1U, 2U, 3U), PermutationVector(1U, 0U, 2U, 3U),
                             PermutationVector(2U, 0U, 1U, 3U),
                             PermutationVector(0U, 2U, 1U, 3U), PermutationVector(1U, 2U, 0U, 3U),
                             PermutationVector(2U, 1U, 0U, 3U),
                             PermutationVector(2U, 1U, 3U, 0U), PermutationVector(1U, 2U, 3U, 0U),
                             PermutationVector(3U, 2U, 1U, 0U),
                             PermutationVector(2U, 3U, 1U, 0U), PermutationVector(1U, 3U, 2U, 0U),
                             PermutationVector(3U, 1U, 2U, 0U),
                             PermutationVector(3U, 0U, 2U, 1U), PermutationVector(0U, 3U, 2U, 1U),
                             PermutationVector(2U, 3U, 0U, 1U),
                             PermutationVector(3U, 2U, 0U, 1U), PermutationVector(0U, 2U, 3U, 1U),
                             PermutationVector(2U, 0U, 3U, 1U),
                             PermutationVector(1U, 0U, 3U, 2U), PermutationVector(0U, 1U, 3U, 2U),
                             PermutationVector(3U, 1U, 0U, 2U),
                             PermutationVector(1U, 3U, 0U, 2U), PermutationVector(0U, 3U, 1U, 2U),
                             PermutationVector(3U, 0U, 1U, 2U)}};

                    return (permutations2.end() != std::find(permutations2.begin(), permutations2.end(), v)) ||
                           (permutations3.end() != std::find(permutations3.begin(), permutations3.end(), v)) ||
                           (permutations4.end() != std::find(permutations4.begin(), permutations4.end(), v));
                }

                BIStatus
                validate_arguments(const BIITensorInfo *src, const BIITensorInfo *dst, const PermutationVector &perm) {
                    BI_COMPUTE_RETURN_ERROR_ON(src->data_type() == BIDataType::UNKNOWN);
                    BI_COMPUTE_RETURN_ERROR_ON_MSG(!is_permutation_supported(perm),
                                                   "PermutationVector not supported.");

                    const BITensorShape dst_shape = misc::shape_calculator::compute_permutation_output_shape(*src,
                                                                                                             perm);

                    // Validate configured destination
                    if (dst->total_size() != 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), dst_shape);
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
                    }

                    return BIStatus{};
                }

                template<typename T>
                void run_permute(const BIWindow &window, const BIITensor *src, const BIITensor *dst,
                                 const PermutationVector &perm) {
                    // Source window
                    BIWindow window_src = window;

                    // we only support these two configs in src/core/NEON/kernels/convolution/common/shims.hpp, for all others
                    // we have to fall back to C++
                    if (perm == PermutationVector{2U, 0U, 1U} || perm == PermutationVector{1U, 2U, 0U}) {
                        window_src.set(BIWindow::DimX,
                                       BIWindow::BIDimension(window.x().start(), window.x().end(),
                                                             window.x().end() - window.x().start()));
                        window_src.set(BIWindow::DimY,
                                       BIWindow::BIDimension(window.y().start(), window.y().end(),
                                                             window.y().end() - window.y().start()));
                        window_src.set(BIWindow::DimZ,
                                       BIWindow::BIDimension(window.z().start(), window.z().end(),
                                                             window.z().end() - window.z().start()));
                        window_src.set(3, BIWindow::BIDimension(window[3].start(), window[3].end(),
                                                                window[3].end() - window[3].start()));
                    }

                    // Destination window
                    BIWindow window_dst(window);
                    const BIWindow::BIDimension zero_window = BIWindow::BIDimension(0, 0, 0);
                    for (size_t d = 0; d <= dst->info()->num_dimensions(); ++d) {
                        window_dst.set(d, zero_window);
                    }

                    // Create iterators
                    BIIterator src_it(src, window_src);
                    BIIterator dst_it(dst, window_dst);

                    // CHW -> HWC
                    if (perm == PermutationVector{2U, 0U, 1U}) {
                        const int in_row_stride = src->info()->strides_in_bytes().y() / sizeof(T);
                        const int in_channel_stride = src->info()->strides_in_bytes().z() / sizeof(T);
                        const int in_batch_stride = src->info()->strides_in_bytes()[3] / sizeof(T);
                        const int n_cols = src->info()->tensor_shape().x();
                        const int n_rows = window_src.y().step();
                        const int n_channels = src->info()->tensor_shape().z();
                        const int n_batches = src->info()->tensor_shape()[3];
                        const int out_channel_stride = dst->info()->strides_in_bytes().x() / sizeof(T);
                        const int out_col_stride = dst->info()->strides_in_bytes().y() / sizeof(T);
                        const int out_row_stride = dst->info()->strides_in_bytes().z() / sizeof(T);
                        const int out_batch_stride = dst->info()->strides_in_bytes()[3] / sizeof(T);
                        execute_window_loop(
                                window_src,
                                [&](const BICoordinates &id) {
                                    const int idx = id[0] * out_col_stride + id[1] * out_row_stride +
                                                    id[2] * out_channel_stride;
                                    reorder::nchw_to_nhwc(reinterpret_cast<const T *>(src_it.ptr()),
                                                          reinterpret_cast<T *>(dst_it.ptr()) + idx, n_batches,
                                                          n_channels, n_rows, n_cols,
                                                          in_batch_stride, in_channel_stride, in_row_stride,
                                                          out_batch_stride,
                                                          out_row_stride, out_col_stride);
                                },
                                src_it, dst_it);
                    }
                        // HWC -> CHW
                    else if (perm == PermutationVector{1U, 2U, 0U}) {
                        const int in_col_stride = src->info()->strides_in_bytes().y() / sizeof(T);
                        const int in_row_stride = src->info()->strides_in_bytes().z() / sizeof(T);
                        const int in_batch_stride = src->info()->strides_in_bytes()[3] / sizeof(T);
                        const int n_channels = src->info()->tensor_shape().x();
                        const int n_cols = window_src.y().step();
                        const int n_rows = src->info()->tensor_shape().z();
                        const int n_batches = src->info()->tensor_shape()[3];
                        const int out_col_stride = dst->info()->strides_in_bytes().x() / sizeof(T);
                        const int out_row_stride = dst->info()->strides_in_bytes().y() / sizeof(T);
                        const int out_channel_stride = dst->info()->strides_in_bytes().z() / sizeof(T);
                        const int out_batch_stride = dst->info()->strides_in_bytes()[3] / sizeof(T);
                        execute_window_loop(
                                window_src,
                                [&](const BICoordinates &id) {
                                    const int idx = id[0] * out_channel_stride + id[1] * out_col_stride +
                                                    id[2] * out_row_stride;
                                    reorder::nhwc_to_nchw(reinterpret_cast<const T *>(src_it.ptr()),
                                                          reinterpret_cast<T *>(dst_it.ptr()) + idx, n_batches, n_rows,
                                                          n_cols, n_channels,
                                                          in_batch_stride, in_row_stride, in_col_stride,
                                                          out_batch_stride,
                                                          out_channel_stride, out_row_stride);
                                },
                                src_it, dst_it);
                    } else {
                        // All other cases fall back to C++
                        // Permute strides
                        BIStrides strides = dst->info()->strides_in_bytes();
                        BIStrides perm_strides = strides;
                        permute_strides(perm_strides, perm);
                        const int perm_stride_3 = src->info()->num_dimensions() >= 4 ? perm_strides[3] : 0;
                        execute_window_loop(
                                window,
                                [&](const BICoordinates &id) {
                                    const int idx =
                                            id[0] * perm_strides[0] + id[1] * perm_strides[1] +
                                            id[2] * perm_strides[2] + id[3] * perm_stride_3;
                                    *(reinterpret_cast<T *>(dst_it.ptr() +
                                                            idx)) = *(reinterpret_cast<const T *>(src_it.ptr()));
                                },
                                src_it, dst_it);
                    }
                }
            } // namespace

            void
            BICpuPermuteKernel::configure(const BIITensorInfo *src, BIITensorInfo *dst, const PermutationVector &perm) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                const BITensorShape dst_shape = misc::shape_calculator::compute_permutation_output_shape(*src, perm);
                // Destination auto inizialitation if not yet initialized
                auto_init_if_empty(*dst, src->clone()->set_tensor_shape(dst_shape));

                // Perform validation step
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, perm));

                _perm = perm;

                // Configure kernel window
                BIWindow win = calculate_max_window(*src, BISteps());

                // This kernel doesn't need padding so update_window_and_padding() can be skipped

                BIICpuKernel::configure(win);
            }

            BIStatus
            BICpuPermuteKernel::validate(const BIITensorInfo *src, const BIITensorInfo *dst,
                                         const PermutationVector &perm) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, perm));
                return BIStatus{};
            }

            void BICpuPermuteKernel::run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                const auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                switch (src->info()->element_size()) {
                    case 1:
                        run_permute<uint8_t>(window, src, dst, _perm);
                        break;
                    case 2:
                        run_permute<uint16_t>(window, src, dst, _perm);
                        break;
                    case 4:
                        run_permute<uint32_t>(window, src, dst, _perm);
                        break;
                    default:
                        BI_COMPUTE_ERROR("Element size not supported");
                        break;
                }
            }

            const char *BICpuPermuteKernel::name() const {
                return "BICpuPermuteKernel";
            }
        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer