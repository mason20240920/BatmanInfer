//
// Created by Mason on 2025/1/9.
//

#include <cpu/kernels/bi_cpu_gemm_matrix_multiply_kernel.hpp>
#include <cpu/kernels/gemm_matrix_mul/list.hpp>
#include <data/core/bi_helpers.hpp>
#include <common/bi_registers.hpp>
#include "data/core/cpp/bi_cpp_validate.hpp"
#include "data/core/helpers/bi_auto_configuration.hpp"
#include <data/core/bi_tensor_info.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                static const std::vector<BICpuGemmMatrixMultiplyKernel::BIGemmMatrixMulKernel> available_kernels = {
                        {"neon_fp32_gemm_matrix_mul", [](const BIDataTypeISASelectorData &data) {
                            return (data.dt == BIDataType::F32);
                        },
                                REGISTER_FP32_NEON(neon_fp32_gemm_matrix_mul)},
                        {"neon_fp16_gemm_matrix_mul",
                                                      [](const BIDataTypeISASelectorData &data) {
                                                          return (data.dt == BIDataType::F16) && data.isa.fp16;
                                                      },
                                REGISTER_FP16_NEON(neon_fp16_gemm_matrix_mul)},
                };

                inline BIStatus validate_arguments(const BIITensorInfo *lhs,
                                                   const BIITensorInfo *rhs,
                                                   const BIITensorInfo *dst,
                                                   float alpha,
                                                   bool is_interleaved,
                                                   const BIGemmReshapeInfo &reshape_info) {
                    BI_COMPUTE_UNUSED(alpha);

                    BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(lhs);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, BIDataType::F16, BIDataType::F32);
                    BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, rhs, dst);

                    if (!is_interleaved) {
                        BI_COMPUTE_RETURN_ERROR_ON(lhs->dimension(0) != rhs->dimension(1));

                        if (dst->total_size() != 0) {
                            BI_COMPUTE_RETURN_ERROR_ON(rhs->dimension(0) != dst->dimension(0));
                            BI_COMPUTE_RETURN_ERROR_ON(lhs->dimension(1) != dst->dimension(1));
                            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);
                        }
                    } else {
                        const int m = reshape_info.m();
                        const int n = reshape_info.n();
                        const int k = reshape_info.k();
                        const int multi_transpose1xW_width = reshape_info.multi_transpose1xW_width();
                        const int multi_interleave4x4_height = reshape_info.multi_interleave4x4_height();

                        /* Interleave */
                        BITensorShape tensor_shape0{lhs->tensor_shape()};
                        tensor_shape0.set(0, k);
                        tensor_shape0.set(1, m);

                        const BITensorInfo tensor_info0 = lhs->clone()->set_tensor_shape(tensor_shape0);
                        const BITensorInfo tensor_info_reshaped0 = lhs->clone()->set_tensor_shape(
                                misc::shape_calculator::compute_interleaved_shape(tensor_info0,
                                                                                  multi_interleave4x4_height));
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(lhs, &tensor_info_reshaped0);

                        if (n != 0) /* Transpose */
                        {
                            BITensorShape tensor_shape1{rhs->tensor_shape()};
                            tensor_shape1.set(0, n);
                            tensor_shape1.set(1, k);

                            const BITensorInfo tensor_info1 = rhs->clone()->set_tensor_shape(tensor_shape1);
                            const BITensorInfo tensor_info_reshaped1 =
                                    rhs->clone()->set_tensor_shape(
                                            misc::shape_calculator::compute_transpose_1xw_with_element_size_shape(
                                                    tensor_info1, multi_transpose1xW_width));
                            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(rhs, &tensor_info_reshaped1);
                        }

                        if (dst->total_size() != 0) {
                            if (n != 0) {
                                BI_COMPUTE_RETURN_ERROR_ON(dst->dimension(0) != static_cast<size_t>(n));
                            }
                            BI_COMPUTE_RETURN_ERROR_ON(dst->dimension(1) != static_cast<size_t>(m));
                            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);
                        }
                    }

                    return BIStatus{};
                }

            } // namespace

            void BICpuGemmMatrixMultiplyKernel::configure(const BIITensorInfo *lhs,
                                                          const BIITensorInfo *rhs,
                                                          BIITensorInfo *dst, float alpha,
                                                          bool is_interleaved,
                                                          const BIGemmReshapeInfo &reshape_info) {
                BI_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);

                BITensorShape tensor_shape{lhs->tensor_shape()};
                tensor_shape.set(0, is_interleaved ? reshape_info.n() : rhs->dimension(0));
                tensor_shape.set(1, is_interleaved ? reshape_info.m() : lhs->dimension(1));

                auto_init_if_empty(*dst, lhs->clone()->set_tensor_shape(tensor_shape));

                // Perform validate step
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(lhs, rhs, dst, alpha, is_interleaved, reshape_info));

                _alpha = alpha;

                // Configure kernel window
                BIWindow win{};

                // Check if the dst tensor is a vector. If so,the kernel runs the vector-matrix multiplication
                const bool is_dst_vector = (dst->dimension(1) == 1);
                if (is_dst_vector) {
                    const unsigned int num_elems_processed_per_iteration_x = (lhs->data_type() == BIDataType::F32) ? 16
                                                                                                                   : 32;

                    win = calculate_max_window(*dst, BISteps(num_elems_processed_per_iteration_x));
                } else {
                    constexpr unsigned int num_elems_processed_per_iteration_x = 8;
                    constexpr unsigned int num_elems_processed_per_iteration_y = 4;

                    win =
                            calculate_max_window(*dst, BISteps(num_elems_processed_per_iteration_x,
                                                               num_elems_processed_per_iteration_y));
                }

                const auto uk = BICpuGemmMatrixMultiplyKernel::get_implementation(
                        BIDataTypeISASelectorData{lhs->data_type(), CPUInfo::get().get_isa()});
                BI_COMPUTE_ERROR_ON_NULLPTR(uk);
                _func = uk->ukernel;

                BIICPPKernel::configure(win);
            }

            BIStatus BICpuGemmMatrixMultiplyKernel::validate(const BIITensorInfo *lhs,
                                                             const BIITensorInfo *rhs,
                                                             const BIITensorInfo *dst, float alpha,
                                                             bool is_interleaved,
                                                             const BIGemmReshapeInfo &reshape_info) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(lhs, rhs, dst, alpha, is_interleaved, reshape_info));

                return BIStatus{};
            }

            void BICpuGemmMatrixMultiplyKernel::run_op(BIITensorPack &tensors,
                                                       const BIWindow &window,
                                                       const ThreadInfo &info) {
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIIKernel::window(), window);
                BI_COMPUTE_ERROR_ON(tensors.empty());
                BI_COMPUTE_ERROR_ON(_func == nullptr);

                const BIITensor *lhs = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
                const BIITensor *rhs = tensors.get_const_tensor(BITensorType::ACL_SRC_1);
                BIITensor *dst = tensors.get_tensor(BITensorType::ACL_DST);

                const bool is_dst_vector = (dst->info()->dimension(1) == 1);
                (*_func)(lhs, rhs, dst, window, info, _alpha, is_dst_vector);
            }

            const char *BICpuGemmMatrixMultiplyKernel::name() const {
                return "BICpuGemmMatrixMultiplyKernel";
            }

            const std::vector<BICpuGemmMatrixMultiplyKernel::BIGemmMatrixMulKernel> &
            BICpuGemmMatrixMultiplyKernel::get_available_kernels() {
                return available_kernels;
            }
        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer