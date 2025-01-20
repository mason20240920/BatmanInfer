//
// Created by Mason on 2025/1/20.
//

#include <cpu/kernels/bi_cpu_gemm_lowp_matrix_reduction_kernel.hpp>

#include <data/core/bi_i_tensor.hpp>
#include <data/core/kernel_descriptors.hpp>
#include <data/core/bi_tensor_info.hpp>

#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>


namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                BIStatus validate_arguments_matrix_a_reduction(const BIITensorInfo *src,
                                                               const BIITensorInfo *dst,
                                                               const GEMMLowpReductionKernelInfo &info) {
                    BI_COMPUTE_UNUSED(info);
                    BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
                    BI_COMPUTE_ERROR_ON_MSG(info.is_reshaped == true, "Not supported");
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::QASYMM8,
                                                                        BIDataType::QASYMM8_SIGNED,
                                                                        BIDataType::QSYMM8,
                                                                        BIDataType::QSYMM8_PER_CHANNEL);

                    if (dst->total_size() > 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, BIDataType::S32);
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                                dst->dimension(0) != src->dimension(1),
                                "Output vector must have length equal to the number of rows of the input matrix");
                    }
                    return BIStatus{};
                }

                BIStatus validate_arguments_matrix_b_reduction(const BIITensorInfo *src,
                                                               const BIITensorInfo *dst,
                                                               const GEMMLowpReductionKernelInfo &info) {
                    BI_COMPUTE_UNUSED(info);
                    BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
                    BI_COMPUTE_ERROR_ON_MSG(info.is_reshaped == true, "Not supported");
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::QASYMM8,
                                                                        BIDataType::QASYMM8_SIGNED,
                                                                        BIDataType::QSYMM8,
                                                                        BIDataType::QSYMM8_PER_CHANNEL);

                    if (dst->total_size() > 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, BIDataType::S32);
                        BI_COMPUTE_RETURN_ERROR_ON_MSG(
                                dst->dimension(0) != src->dimension(0),
                                "Output vector must have length equal to the number of columns of the input matrix");
                    }
                    return BIStatus{};
                }
            }

            void BICpuGemmLowpMatrixAReductionKernel::configure(const BatmanInfer::BIITensorInfo *src,
                                                                BatmanInfer::BIITensorInfo *dst,
                                                                const BatmanInfer::GEMMLowpReductionKernelInfo &info) {
                // Perform validate step
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_a_reduction(src, dst, info));
                _k = info.k;
                _scalar = info.scalar;
                _mul_by_scalar = info.mul_by_scalar;

                switch (src->data_type()) {
                    case BIDataType::QASYMM8:
                        _func = &BICpuGemmLowpMatrixAReductionKernel::run_internal<uint8_t>;
                        break;
                    case BIDataType::QASYMM8_SIGNED:
                    case BIDataType::QSYMM8:
                    case BIDataType::QSYMM8_PER_CHANNEL:
                        _func = &BICpuGemmLowpMatrixAReductionKernel::run_internal<int8_t>;
                        break;
                    default:
                        BI_COMPUTE_ERROR("Unsupported data type");
                }

                // Output auto initialization if not yet initialized
                auto_init_if_empty(*dst, BITensorShape(src->dimension(1)), 1, BIDataType::S32);

                BIWindow win = calculate_max_window(*dst, BISteps(1));
                BIICpuKernel::configure(win);
            }

            BIStatus BICpuGemmLowpMatrixAReductionKernel::validate(const BatmanInfer::BIITensorInfo *src,
                                                                   const BatmanInfer::BIITensorInfo *dst,
                                                                   const BatmanInfer::GEMMLowpReductionKernelInfo &info) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_a_reduction(src, dst, info));
                return BIStatus{};
            }

            template<typename T>
            void BICpuGemmLowpMatrixAReductionKernel::run_internal(const BIITensor *src,
                                                                   BIITensor *dst,
                                                                   const BatmanInfer::BIWindow &window) {
                // Intermediate and final accumulator types
                using TIAcc = wrapper::traits::promote_t<T>;
                using TAcc = wrapper::traits::promote_t<TIAcc>;

                BIWindow collapsed_window = window.collapse_if_possible(BIIKernel::window(), BIWindow::DimY);

                BIWindow win_input(collapsed_window);
                win_input.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                win_input.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                win_input.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));

                BIIterator in(src, win_input);
                BIIterator out(dst, collapsed_window);

                execute_window_loop(
                        collapsed_window,
                        [&](const BICoordinates &id) {
                            auto vsum_row = wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{});
                            TAcc sum_row = 0;

                            const T *matrix_a = reinterpret_cast<const T *>(
                                    (in.ptr() + id.x() * src->info()->strides_in_bytes()[1] +
                                     id.y() * src->info()->strides_in_bytes()[2]));

#if __arm__
                            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_a));
#endif /* __arm__ */

                            int i = 0;
                            // This for loop performs 16 accumulations
                            for (; i <= (_k - 16); i += 16) {
                                const auto a0_d8 = wrapper::vloadq(matrix_a + i);

                                // Partial accumulations in U16
                                const auto tmp_sum0 = wrapper::vaddl(wrapper::vgetlow(a0_d8), wrapper::vgethigh(a0_d8));

                                // Accumulate to U32
                                vsum_row = wrapper::vadd(vsum_row, wrapper::vpaddl(tmp_sum0));
                            }

                            // This for loop performs the leftover accumulations
                            for (; i < _k; ++i) {
                                sum_row += static_cast<TAcc>(matrix_a[i]);
                            }

#if defined(__aarch64__)
                            // Reduction operation available on 64 bits architectures only
                            sum_row += wrapper::vaddv(vsum_row);
#else  // __aarch64__
                            auto tmp = wrapper::vpadd(wrapper::vgethigh(vsum_row), wrapper::vgetlow(vsum_row));
            tmp      = wrapper::vpadd(tmp, tmp);

            sum_row += wrapper::vgetlane(tmp, 0);
#endif // __aarch64__

                            // Multiply by scalar if necessary
                            if (_mul_by_scalar) {
                                sum_row *= _scalar;
                            }

                            *(reinterpret_cast<int *>(out.ptr())) = static_cast<int32_t>(sum_row);
                        },
                        in, out);
            }

            void BICpuGemmLowpMatrixAReductionKernel::run_op(BatmanInfer::BIITensorPack &tensors,
                                                             const BatmanInfer::BIWindow &window,
                                                             const BatmanInfer::ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                (this->*_func)(src, dst, window);
            }

            const char *BICpuGemmLowpMatrixAReductionKernel::name() const {
                return "BICpuGemmLowpMatrixAReductionKernel";
            }

            void BICpuGemmLowpMatrixBReductionKernel::configure(const BatmanInfer::BIITensorInfo *src,
                                                                BatmanInfer::BIITensorInfo *dst,
                                                                const BatmanInfer::GEMMLowpReductionKernelInfo &info) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_b_reduction(src, dst, info));

                _k = info.k;
                _scalar = info.scalar;
                _mul_by_scalar = info.mul_by_scalar;

                // Configure kernel window
                constexpr unsigned int num_elems_processed_per_iteration = 16;

                switch (src->data_type()) {
                    case BIDataType::QASYMM8:
                        _func = &BICpuGemmLowpMatrixBReductionKernel::run_internal<uint8_t>;
                        break;
                    case BIDataType::QASYMM8_SIGNED:
                    case BIDataType::QSYMM8:
                    case BIDataType::QSYMM8_PER_CHANNEL:
                        _func = &BICpuGemmLowpMatrixBReductionKernel::run_internal<int8_t>;
                        break;
                    default:
                        BI_COMPUTE_ERROR("Unsupported data type");
                }

                // Output auto initialization if not yet initialized
                auto_init_if_empty(*dst, BITensorShape(src->dimension(0)), 1, BIDataType::S32);

                // Configure kernel window
                BIWindow win = calculate_max_window_horizontal(*dst, BISteps(num_elems_processed_per_iteration));
                BIICpuKernel::configure(win);
            }

            BIStatus BICpuGemmLowpMatrixBReductionKernel::validate(const BIITensorInfo *src,
                                                                   const BIITensorInfo *dst,
                                                                   const GEMMLowpReductionKernelInfo &info) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_b_reduction(src, dst, info));
                return BIStatus{};
            }

            template<typename T>
            void BICpuGemmLowpMatrixBReductionKernel::run_internal(const BIITensor *src,
                                                                   BIITensor *dst,
                                                                   const BIWindow &window,
                                                                   const ThreadInfo &info) {
                // Intermediate and final accumulator types
                using TIAcc = wrapper::traits::promote_t<T>;
                using TAcc = wrapper::traits::promote_t<TIAcc>;

                BIWindow collapsed_window = window.collapse_if_possible(BIIKernel::window(), BIWindow::DimY);
                const auto vec_scalar = wrapper::vdup_n(static_cast<TAcc>(_scalar), wrapper::traits::vector_128_tag{});

                const auto width_matrix_b = static_cast<int>(src->info()->dimension(0));
                const auto in_b_stride = static_cast<int>(src->info()->strides_in_bytes()[1]);

                // The implementation computes 16 elements per iteration
                const int window_start_x = 16 * info.thread_id;
                const int window_step_x = 16 * info.num_threads;
                // Make sure (window_end_x - window_start_x) is a multiple of window_step_x
                const int window_end_x =
                        ceil_to_multiples(width_matrix_b - window_start_x, window_step_x) + window_start_x;

                BIWindow win_out(collapsed_window);
                win_out.set(BIWindow::DimX, BIWindow::BIDimension(window_start_x, window_end_x, window_step_x));

                BIWindow win_in(win_out);
                win_in.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                win_in.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));

                BIIterator inb(src, win_in);
                BIIterator out(dst, win_out);

                execute_window_loop(
                        win_out,
                        [&](const BICoordinates &id) {
                            if (id.x() > width_matrix_b) {
                                return;
                            }

                            // Note: Since the input is unsigned char, we can safely use unsigned int for the accumulation
                            // 4 x u/int32x4_t = 16 column accumulators
                            typename wrapper::traits::neon_bitvector<TAcc, wrapper::traits::BitWidth::W128>::type sum_col[4] = {
                                    wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                                    wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                                    wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                                    wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{})};

                            const auto *matrix_b = reinterpret_cast<const T *>(inb.ptr() + id.y() *
                                                                                           src->info()->strides_in_bytes()[2]);

#if __arm__
                            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_b));
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_b + in_b_stride));
#endif /* __arm__ */

                            // If we have less than 16 columns left, we can't use the main unrolled loop
                            if ((width_matrix_b - id.x()) >= 16) {
                                // Row index
                                int i = 0;
                                // 4 x u/int32x4_t = 16 columns unrolled across 4 rows
                                for (; i <= (_k - 4); i += 4) {
                                    // Load 4 rows of 16 columns of 8bit elements
                                    // (|                   |        )
                                    // (|                   |        )
                                    // (|                   |        )
                                    // (|                   |        )
                                    const auto b0_u8 = wrapper::vloadq(matrix_b + 0 * in_b_stride);
                                    const auto b1_u8 = wrapper::vloadq(matrix_b + 1 * in_b_stride);
                                    const auto b2_u8 = wrapper::vloadq(matrix_b + 2 * in_b_stride);
                                    const auto b3_u8 = wrapper::vloadq(matrix_b + 3 * in_b_stride);

#if __arm__
                                    asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 1 * in_b_stride));
                    asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 2 * in_b_stride));
                    asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 3 * in_b_stride));
                    asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 4 * in_b_stride));
#endif /* __arm__ */

                                    // Partial accumulation to 16bit (4 rows => 2 rows)
                                    // (|         |         |        )
                                    // (|         |         |        )
                                    typename wrapper::traits::neon_bitvector<TIAcc, wrapper::traits::BitWidth::W128>::type tmp_sum[2] =
                                            {wrapper::vdup_n(static_cast<TIAcc>(0), wrapper::traits::vector_128_tag{}),
                                             wrapper::vdup_n(static_cast<TIAcc>(0), wrapper::traits::vector_128_tag{})};

                                    tmp_sum[0] = wrapper::vaddw(tmp_sum[0], wrapper::vgetlow(b1_u8));
                                    tmp_sum[0] = wrapper::vaddw(tmp_sum[0], wrapper::vgetlow(b0_u8));
                                    tmp_sum[0] = wrapper::vaddw(tmp_sum[0], wrapper::vgetlow(b2_u8));
                                    tmp_sum[0] = wrapper::vaddw(tmp_sum[0], wrapper::vgetlow(b3_u8));
                                    tmp_sum[1] = wrapper::vaddw(tmp_sum[1], wrapper::vgethigh(b0_u8));
                                    tmp_sum[1] = wrapper::vaddw(tmp_sum[1], wrapper::vgethigh(b1_u8));
                                    tmp_sum[1] = wrapper::vaddw(tmp_sum[1], wrapper::vgethigh(b2_u8));
                                    tmp_sum[1] = wrapper::vaddw(tmp_sum[1], wrapper::vgethigh(b3_u8));

                                    // Accumulate to 32bit (2 rows => 1 row)
                                    // (|    |    |    |    |        )
                                    sum_col[0] = wrapper::vaddw(sum_col[0], wrapper::vgetlow(tmp_sum[0]));
                                    sum_col[1] = wrapper::vaddw(sum_col[1], wrapper::vgethigh(tmp_sum[0]));
                                    sum_col[2] = wrapper::vaddw(sum_col[2], wrapper::vgetlow(tmp_sum[1]));
                                    sum_col[3] = wrapper::vaddw(sum_col[3], wrapper::vgethigh(tmp_sum[1]));

                                    matrix_b += 4 * in_b_stride;
                                }

                                // This for loop accumulates the rows left over from the 4x unrolling above
                                for (; i < _k; ++i) {
                                    const auto b0_b8 = wrapper::vloadq(matrix_b + 0 * in_b_stride);

                                    // Convert 8bit => 16bit
                                    const typename wrapper::traits::neon_bitvector<TIAcc, wrapper::traits::BitWidth::W128>::type
                                            b0_b16[2]{wrapper::vmovl(wrapper::vgetlow(b0_b8)),
                                                      wrapper::vmovl(wrapper::vgethigh(b0_b8))};

                                    // Accumulate to 32bit
                                    sum_col[0] = wrapper::vaddw(sum_col[0], wrapper::vgetlow(b0_b16[0]));
                                    sum_col[1] = wrapper::vaddw(sum_col[1], wrapper::vgethigh(b0_b16[0]));
                                    sum_col[2] = wrapper::vaddw(sum_col[2], wrapper::vgetlow(b0_b16[1]));
                                    sum_col[3] = wrapper::vaddw(sum_col[3], wrapper::vgethigh(b0_b16[1]));

                                    matrix_b += in_b_stride;
                                }
                            } else {
                                // Accumulate left over columns to sum_cols
                                for (int i = 0; i < _k; ++i) // row loop
                                {
                                    auto left_over_cols = width_matrix_b - id.x();
                                    auto l = left_over_cols;
                                    for (auto k = 0; k < 4 && l; ++k) {
                                        for (auto j = 0; j < 4 && l; ++j, --l) {
                                            sum_col[k][j] += matrix_b[left_over_cols - l];
                                        }
                                    }
                                    matrix_b += in_b_stride;
                                }
                            }

                            // Multiply by scalar if necessary
                            if (_mul_by_scalar) {
                                sum_col[0] = wrapper::vmul(sum_col[0], vec_scalar);
                                sum_col[1] = wrapper::vmul(sum_col[1], vec_scalar);
                                sum_col[2] = wrapper::vmul(sum_col[2], vec_scalar);
                                sum_col[3] = wrapper::vmul(sum_col[3], vec_scalar);
                            }

                            auto vector_sum_col = reinterpret_cast<int32_t *>(out.ptr());
                            if ((width_matrix_b - id.x()) >= 16) {
                                wrapper::vstore(vector_sum_col + 0, wrapper::vreinterpret(sum_col[0]));
                                wrapper::vstore(vector_sum_col + 4, wrapper::vreinterpret(sum_col[1]));
                                wrapper::vstore(vector_sum_col + 8, wrapper::vreinterpret(sum_col[2]));
                                wrapper::vstore(vector_sum_col + 12, wrapper::vreinterpret(sum_col[3]));
                            } else {
                                auto left_over = width_matrix_b - id.x();
                                for (auto k = 0; k < 4 && left_over; ++k) {
                                    for (auto j = 0; j < 4 && left_over; ++j, --left_over) {
                                        *(vector_sum_col + k * 4 + j) = sum_col[k][j];
                                    }
                                }
                            }
                        },
                        inb, out);
            }

            void BICpuGemmLowpMatrixBReductionKernel::run_op(BIITensorPack &tensors, const BIWindow &window,
                                                             const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                (this->*_func)(src, dst, window, info);
            }

            const char *BICpuGemmLowpMatrixBReductionKernel::name() const {
                return "BICpuGemmLowpMatrixBReductionKernel";
            }

        } // namespace kernels
    } // namespace cpu
}