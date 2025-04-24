//
// Created by Mason on 2025/2/7.
//
#include <cpu/kernels/bi_cpu_gemm_lowp_quantize_down_int32_to_int8_scale_by_fixed_point_kernel.hpp>

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
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, BIDataType::QASYMM8_SIGNED);
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, src);
                    }

                    return BIStatus{};
                }
            } // namespace

            /**
             * @brief 进行GemmLowp scale 定点数运算
             * @tparam is_bounded_relu
             * @param src 输入张量信息
             * @param bias 输入张量偏置值
             * @param dst 输入张量的结果位置
             * @param window 遍历窗口
             */
            template<bool is_bounded_relu>
            void BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::run_internal(const BIITensor *src,
                const BIITensor *bias,
                BIITensor *dst,
                const BIWindow &window) {
                const int32x4_t result_offset_after_shift_s32 = vdupq_n_s32(
                    _result_offset_after_shift);
                const int8x16_t min_s8 = vdupq_n_s8(static_cast<int8_t>(_min));
                const int8x16_t max_s8 = vdupq_n_s8(static_cast<int8_t>(_max));
                BI_COMPUTE_UNUSED(min_s8, max_s8);
                // 一次性处理16个元素 (SIMD) 并行
                // 如果是per channel 量化
                constexpr int window_step_x = 16;
                const auto window_start_x = static_cast<int>(window.x().start());
                const auto window_end_x = static_cast<int>(window.x().end());

                // 合并z维窗口，x维每次只处理1个元素（方便SIMD）
                BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
                win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                BIIterator in(src, win_collapsed);
                BIIterator out(dst, win_collapsed);
                if (_is_per_channel) {
                    if (bias != nullptr) {
                        BIWindow win_biases; // 如果有bias，准备bias迭代器
                        win_biases.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                        win_biases.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));
                        BIIterator bias_i(bias, win_biases);
                        // 主循环，遍历所有窗口
                        execute_window_loop(
                            win_collapsed,
                            [&](const BICoordinates &coord) {
                                // Compute 16 elements per iteration
                                int x = window_start_x;
                                // Compute left-over elements
                                for (; x < window_end_x; ++x) {
                                    const int32_t bias_value = *(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x);
                                    int32_t in_value = *(reinterpret_cast<const int32_t *>(in.ptr()) + x);
                                    // Add bias
                                    in_value += bias_value;
                                    // Finalize and store the result
                                    *reinterpret_cast<int8_t *>(out.ptr() + x) = finalize_quantization(
                                        in_value, _result_fixedpoint_multiplier, _result_shift,
                                        _result_offset_after_shift,
                                        static_cast<int8_t>(_min), static_cast<int8_t>(_max), is_bounded_relu);
                                }
                            },
                            in, out, bias_i);
                    } else {
                        execute_window_loop(
                            win_collapsed,
                            [&](const BICoordinates &) {
                                // Compute 16 elements per iteration
                                int x = window_start_x;
                                // Compute left-over elements
                                for (; x < window_end_x; ++x) {
                                    const int32_t in_value = *(reinterpret_cast<const int32_t *>(in.ptr()) + x);
                                    const auto curr_result_fixedpoint_multiplier = _result_fixedpoint_multipliers[x];
                                    const auto curr_result_shift = _result_shifts[x];

                                    // Finalize and store the result
                                    *reinterpret_cast<int8_t *>(out.ptr() + x) = finalize_quantization(
                                        in_value, curr_result_fixedpoint_multiplier, curr_result_shift,
                                        _result_offset_after_shift,
                                        static_cast<int8_t>(_min), static_cast<int8_t>(_max), is_bounded_relu);
                                }
                            },
                            in, out);
                    }
                    return;
                }


                // 预先broadcast好的一些常量
                if (bias != nullptr) {
                    BIWindow win_biases; // 如果有bias，准备bias迭代器
                    win_biases.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                    win_biases.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));
                    BIIterator bias_i(bias, win_biases);

                    // 主循环，遍历所有窗口
                    execute_window_loop(
                        win_collapsed,
                        [&](const BICoordinates &coord) {
                            // Compute 16 elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                // 载入16个int32输入和bias（4x4向量）
                                int32x4x4_t in_s32 = {
                                    {
                                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 0),
                                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 4),
                                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 8),
                                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 12)
                                    }
                                };
                                const int32x4x4_t bias_s32 = {
                                    {
                                        vld1q_s32(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x + 0),
                                        vld1q_s32(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x + 4),
                                        vld1q_s32(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x + 8),
                                        vld1q_s32(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x + 12)
                                    }
                                };
                                // Add the bias to GEMM's result
                                in_s32.val[0] = vaddq_s32(in_s32.val[0], bias_s32.val[0]);
                                in_s32.val[1] = vaddq_s32(in_s32.val[1], bias_s32.val[1]);
                                in_s32.val[2] = vaddq_s32(in_s32.val[2], bias_s32.val[2]);
                                in_s32.val[3] = vaddq_s32(in_s32.val[3], bias_s32.val[3]);
                                // 量化并写回int8结果
                                vst1q_s8(reinterpret_cast<int8_t *>(out.ptr() + x),
                                         finalize_quantization(in_s32, _result_fixedpoint_multiplier, _result_shift,
                                             result_offset_after_shift_s32, min_s8, max_s8,
                                             is_bounded_relu));
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                const int32_t bias_value = *(reinterpret_cast<const int32_t *>(bias_i.ptr()) + x);
                                int32_t in_value = *(reinterpret_cast<const int32_t *>(in.ptr()) + x);

                                // Add bias
                                in_value += bias_value;
                                // Finalize and store the result
                                *reinterpret_cast<int8_t *>(out.ptr() + x) = finalize_quantization(
                                    in_value, _result_fixedpoint_multiplier, _result_shift,
                                    _result_offset_after_shift,
                                    static_cast<int8_t>(_min), static_cast<int8_t>(_max), is_bounded_relu);
                            }
                        },
                        in, out, bias_i);
                } else {
                    execute_window_loop(
                        win_collapsed,
                        [&](const BICoordinates &coord) {
                            // Compute 16 elements per iteration
                            int x = window_start_x;
                            for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                int32x4x4_t in_s32 = {
                                    {
                                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 0),
                                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 4),
                                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 8),
                                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 12)
                                    }
                                };

                                vst1q_s8(reinterpret_cast<int8_t *>(out.ptr() + x),
                                         finalize_quantization(in_s32, _result_fixedpoint_multiplier, _result_shift,
                                             result_offset_after_shift_s32, min_s8, max_s8,
                                             is_bounded_relu));
                            }

                            // Compute left-over elements
                            for (; x < window_end_x; ++x) {
                                const int32_t in_value = *(reinterpret_cast<const int32_t *>(in.ptr()) + x);

                                // Finalize and store the result
                                *reinterpret_cast<int8_t *>(out.ptr() + x) = finalize_quantization(
                                    in_value, _result_fixedpoint_multiplier, _result_shift,
                                    _result_offset_after_shift,
                                    static_cast<int8_t>(_min), static_cast<int8_t>(_max), is_bounded_relu);
                            }
                        },
                        in, out);
                }
            }

            void BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::configure(BIITensorInfo *src,
                BIITensorInfo *bias,
                BIITensorInfo *dst,
                const std::vector<int> &result_fixedpoint_multipliers,
                const std::vector<int> &result_shifts,
                int result_fixedpoint_multiplier,
                int result_shift,
                int result_offset_after_shift,
                bool is_per_channel,
                int min,
                int max) {
                BI_COMPUTE_UNUSED(bias);
                // Perform validate step
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src, bias, dst, min, max));

                _result_fixedpoint_multipliers = result_fixedpoint_multipliers;
                _result_shifts = result_shifts;
                _result_fixedpoint_multiplier = result_fixedpoint_multiplier;
                _result_shift = result_shift;
                _result_offset_after_shift = result_offset_after_shift;
                _min = min;
                _max = max;
                _is_per_channel = is_per_channel;

                // Output auto initialization if not yet initialized
                auto_init_if_empty(*dst, src->clone()->set_data_type(BIDataType::QASYMM8_SIGNED));

                // Configure kernel window
                BIWindow win_config = calculate_max_window(*src, BISteps());
                BIICpuKernel::configure(win_config);

                // Check if we need to clamp the result using min and max
                const bool is_bounded_relu = !(min <= -128 && max >= 127);
                _func = is_bounded_relu
                            ? &BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::run_internal<true>
                            : &BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::run_internal<false>;
            }

            void BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::dynamic_configure(BIITensorInfo *src) {
                auto win = BIICpuKernel::window();
                dynamic_calculate_max_window(*src, win);
                BIICpuKernel::dynamic_configure(win);
            }


            BIStatus BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::validate(
                const BIITensorInfo *src, const BIITensorInfo *bias, const BIITensorInfo *dst, int min, int max) {
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, bias, dst, min, max));
                return BIStatus{};
            }

            void BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::run_op(BIITensorPack &tensors,
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

            const char *BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::name() const {
                return "BICpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel";
            }
        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer
