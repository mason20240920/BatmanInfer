//
// Created by Mason on 2025/2/7.
//

#include <cpu/kernels/bi_cpu_gemm_lowp_quantize_down_int32_to_uint8_scale_by_fixed_point_kernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_utils.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/bi_ne_asymm.hpp>

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                BIStatus
                validate_arguments(const BIITensorInfo *src, const BIITensorInfo *bias, const BIITensorInfo *dst,
                                   int min,
                                   int max) {
                    BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::S32);
                    BI_COMPUTE_RETURN_ERROR_ON(min > max);

                    // Check biases if exist
                    if (bias != nullptr) {
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, bias);
                        BI_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
                        BI_COMPUTE_RETURN_ERROR_ON(src->dimension(0) != bias->dimension(0));
                    }

                    if (dst->total_size() != 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, BIDataType::QASYMM8);
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, src);
                    }

                    return BIStatus{};
                }
            } // namespace

            template<bool is_bounded_relu>
            void BICpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::run_internal(const BIITensor *src,
                                                                                            const BIITensor *bias,
                                                                                            BIITensor *dst,
                                                                                            const BIWindow &window) {
                const int32x4_t result_offset_after_shift_s32 = vdupq_n_s32(_result_offset_after_shift);
                const uint8x16_t min_u8 = vdupq_n_u8(static_cast<uint8_t>(_min));
                const uint8x16_t max_u8 = vdupq_n_u8(static_cast<uint8_t>(_max));

                BI_COMPUTE_UNUSED(min_u8);
                BI_COMPUTE_UNUSED(max_u8);

                const int window_step_x = 16;
                const auto window_start_x = static_cast<int>(window.x().start());
                const auto window_end_x = static_cast<int>(window.x().end());

                BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
                win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                BIIterator in(src, win_collapsed);
                BIIterator out(dst, win_collapsed);
                if (bias != nullptr) {
                    BIWindow win_biases;
                    win_biases.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                    win_biases.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));

                    BIIterator bias_i(bias, win_biases);
                    execute_window_loop(
                            win_collapsed,
                            [&](const BICoordinates &) {
                                // Compute 16 elements per iteration
                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                    int32x4x4_t in_s32 = {
                                            {vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 0),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 4),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 8),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 12)}};

                                    const int32x4x4_t bias_s32 = {
                                            {vld1q_s32(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x + 0),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x + 4),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x + 8),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x + 12)}};

                                    // Add the bias to GEMM's result
                                    in_s32.val[0] = vaddq_s32(in_s32.val[0], bias_s32.val[0]);
                                    in_s32.val[1] = vaddq_s32(in_s32.val[1], bias_s32.val[1]);
                                    in_s32.val[2] = vaddq_s32(in_s32.val[2], bias_s32.val[2]);
                                    in_s32.val[3] = vaddq_s32(in_s32.val[3], bias_s32.val[3]);

                                    vst1q_u8(out.ptr() + x,
                                             finalize_quantization(in_s32, _result_fixedpoint_multiplier, _result_shift,
                                                                   result_offset_after_shift_s32, min_u8, max_u8,
                                                                   is_bounded_relu));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x) {
                                    const int32_t bias_value = *(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x);
                                    int32_t in_value = *(reinterpret_cast<const int32_t *>(in.ptr()) + x);

                                    // Add bias
                                    in_value += bias_value;
                                    // Finalize and store the result
                                    *(out.ptr() + x) = finalize_quantization(in_value, _result_fixedpoint_multiplier,
                                                                             _result_shift,
                                                                             _result_offset_after_shift,
                                                                             static_cast<uint8_t>(_min),
                                                                             static_cast<uint8_t>(_max),
                                                                             is_bounded_relu);
                                }
                            },
                            in, out, bias_i);
                } else {
                    execute_window_loop(
                            win_collapsed,
                            [&](const BICoordinates &) {
                                // Compute 16 elements per iteration
                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                    int32x4x4_t in_s32 = {
                                            {vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 0),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 4),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 8),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 12)}};

                                    vst1q_u8(out.ptr() + x,
                                             finalize_quantization(in_s32, _result_fixedpoint_multiplier, _result_shift,
                                                                   result_offset_after_shift_s32, min_u8, max_u8,
                                                                   is_bounded_relu));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x) {
                                    const int32_t in_value = *(reinterpret_cast<const int32_t *>(in.ptr()) + x);

                                    // Finalize and store the result
                                    *(out.ptr() + x) = finalize_quantization(in_value, _result_fixedpoint_multiplier,
                                                                             _result_shift,
                                                                             _result_offset_after_shift,
                                                                             static_cast<uint8_t>(_min),
                                                                             static_cast<uint8_t>(_max),
                                                                             is_bounded_relu);
                                }
                            },
                            in, out);
                }
            }

            void BICpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::configure(BIITensorInfo *src,
                                                                                         BIITensorInfo *bias,
                                                                                         BIITensorInfo *dst,
                                                                                         int result_fixedpoint_multiplier,
                                                                                         int result_shift,
                                                                                         int result_offset_after_shift,
                                                                                         int min,
                                                                                         int max) {
                BI_COMPUTE_UNUSED(bias);
                // Perform validate step
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src, bias, dst, min, max));

                _result_fixedpoint_multiplier = result_fixedpoint_multiplier;
                _result_shift = result_shift;
                _result_offset_after_shift = result_offset_after_shift;
                _min = min;
                _max = max;

                // Output auto inizialitation if not yet initialized
                auto_init_if_empty(*dst, src->clone()->set_data_type(BIDataType::QASYMM8));

                // Configure kernel window
                auto win_config = calculate_max_window(*src, BISteps());
                BIICpuKernel::configure(win_config);

                // Check if we need to clamp the result using min and max
                const bool is_bounded_relu = !(min <= 0 && max >= 255);
                _func = is_bounded_relu
                        ? &BICpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::run_internal<true>
                        : &BICpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::run_internal<false>;
            }

            BIStatus BICpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::validate(
                    const BIITensorInfo *src, const BIITensorInfo *bias, const BIITensorInfo *dst, int min, int max) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, bias, dst, min, max));
                return BIStatus{};
            }

            void BICpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::run_op(BIITensorPack &tensors,
                                                                                      const BIWindow &window,
                                                                                      const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);
                BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");

                auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
                auto bias = tensors.get_const_tensor(BITensorType::ACL_BIAS);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                (this->*_func)(src, bias, dst, window);
            }

            const char *BICpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::name() const {
                return "BICpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel";
            }
        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer
