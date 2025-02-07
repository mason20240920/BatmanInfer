//
// Created by Mason on 2025/2/7.
//

#include <cpu/kernels/bi_cpu_gemm_lowp_quantize_down_int32_scale_kernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_utils.hpp>
#include <data/core/utils/quantization/asymm_helpers.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                BIStatus validate_arguments(const BIITensorInfo *src,
                                            const BIITensorInfo *bias,
                                            const BIITensorInfo *dst,
                                            const BIGEMMLowpOutputStageInfo *output_stage) {
                    BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, BIDataType::S32);

                    BI_COMPUTE_RETURN_ERROR_ON(
                            output_stage->gemmlowp_max_bound >
                            std::get<1>(quantization::get_min_max_values_from_quantized_data_type(
                                    output_stage->output_data_type)));
                    BI_COMPUTE_RETURN_ERROR_ON(
                            output_stage->gemmlowp_min_bound <
                            std::get<0>(quantization::get_min_max_values_from_quantized_data_type(
                                    output_stage->output_data_type)) ||
                            output_stage->gemmlowp_min_bound > output_stage->gemmlowp_max_bound);

                    // Check biases if exist
                    if (bias != nullptr) {
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, bias);
                        BI_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
                        BI_COMPUTE_RETURN_ERROR_ON(src->dimension(0) != bias->dimension(0));
                    }

                    if (dst->total_size() != 0) {
                        if (dst->data_type() != output_stage->output_data_type &&
                            (output_stage->output_data_type == BIDataType::QASYMM8 ||
                             output_stage->output_data_type == BIDataType::QASYMM8_SIGNED)) {
                            BI_COMPUTE_RETURN_ERROR_MSG("Mismatching data types");
                        }

                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
                    }

                    return BIStatus{};
                }

                inline void scale_input(int32x4x4_t &in_s32, int32x4_t result_offset_s32, int32_t result_mult_int) {
                    // Add the offset terms to GEMM's result
                    in_s32.val[0] = vaddq_s32(in_s32.val[0], result_offset_s32);
                    in_s32.val[1] = vaddq_s32(in_s32.val[1], result_offset_s32);
                    in_s32.val[2] = vaddq_s32(in_s32.val[2], result_offset_s32);
                    in_s32.val[3] = vaddq_s32(in_s32.val[3], result_offset_s32);

                    // Multiply by result_mult_int
                    in_s32.val[0] = vmulq_n_s32(in_s32.val[0], result_mult_int);
                    in_s32.val[1] = vmulq_n_s32(in_s32.val[1], result_mult_int);
                    in_s32.val[2] = vmulq_n_s32(in_s32.val[2], result_mult_int);
                    in_s32.val[3] = vmulq_n_s32(in_s32.val[3], result_mult_int);
                }

                template<typename T>
                inline
                typename std::enable_if<std::is_same<T, uint8_t>::value, typename wrapper::traits::neon_vector<T, 16>::type>::type
                convert_to_8bit(const int16x8x2_t in_s16) {
                    return wrapper::vcombine(wrapper::vqmovun(in_s16.val[0]), wrapper::vqmovun(in_s16.val[1]));
                }

                template<typename T>
                inline typename std::enable_if<std::is_same<T, int8_t>::value, typename wrapper::traits::neon_vector<T, 16>::type>::type
                convert_to_8bit(const int16x8x2_t in_s16) {
                    return wrapper::vcombine(wrapper::vqmovn(in_s16.val[0]), wrapper::vqmovn(in_s16.val[1]));
                }

                template<typename T>
                inline typename wrapper::traits::neon_vector<T, 16>::type
                finalize_quantization(int32x4x4_t &in_s32,
                                      int32x4_t result_shift_s32,
                                      typename wrapper::traits::neon_vector<T, 16>::type min,
                                      typename wrapper::traits::neon_vector<T, 16>::type max) {
                    // Shift final result (negative value shift right)
                    in_s32.val[0] = vshlq_s32(in_s32.val[0], result_shift_s32);
                    in_s32.val[1] = vshlq_s32(in_s32.val[1], result_shift_s32);
                    in_s32.val[2] = vshlq_s32(in_s32.val[2], result_shift_s32);
                    in_s32.val[3] = vshlq_s32(in_s32.val[3], result_shift_s32);

                    // Convert S32 to S16
                    const int16x8x2_t in_s16 = {{vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                                                 vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))}};

                    // Convert S16 to S8 or U8
                    typename wrapper::traits::neon_vector<T, 16>::type out = convert_to_8bit<T>(in_s16);

                    out = wrapper::vmax(out, min);
                    out = wrapper::vmin(out, max);

                    return out;
                }
            } // namespace

            template<typename T>
            void BICpuGemmLowpQuantizeDownInt32ScaleKernel::run_internal(const BIITensor *src,
                                                                         const BIITensor *bias,
                                                                         BIITensor *dst,
                                                                         const BIWindow &window) {
                using VectorType = typename wrapper::traits::neon_vector<T, 16>::type;

                const int32x4_t result_offset_s32 = vdupq_n_s32(_output_stage->gemmlowp_offset);
                const int32x4_t result_shift_s32 = vdupq_n_s32(-_output_stage->gemmlowp_shift);
                const int window_step_x = 16;
                const auto window_start_x = static_cast<int>(window.x().start());
                const auto window_end_x = static_cast<int>(window.x().end());

                const int clamp_min = (_is_bounded_relu) ? _output_stage->gemmlowp_min_bound
                                                         : std::numeric_limits<T>::lowest();
                const int clamp_max = (_is_bounded_relu) ? _output_stage->gemmlowp_max_bound
                                                         : std::numeric_limits<T>::max();

                VectorType min = wrapper::vdup_n(static_cast<T>(clamp_min), wrapper::traits::vector_128_tag{});
                VectorType max = wrapper::vdup_n(static_cast<T>(clamp_max), wrapper::traits::vector_128_tag{});

                BIWindow win(window);
                win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                BIIterator in(src, win);
                BIIterator out(dst, win);

                if (bias != nullptr) {
                    BIWindow win_biases;
                    win_biases.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
                    win_biases.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));

                    BIIterator bias_i(bias, win_biases);
                    execute_window_loop(
                            win,
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

                                    // Add the offset terms to GEMM's result and multiply by result_mult_int
                                    scale_input(in_s32, result_offset_s32, _output_stage->gemmlowp_multiplier);

                                    wrapper::vstore(reinterpret_cast<T *>(out.ptr() + x),
                                                    finalize_quantization<T>(in_s32, result_shift_s32, min, max));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x) {
                                    const int bias_value = *(reinterpret_cast<const int *>(bias_i.ptr()) + x);
                                    int in_value = *(reinterpret_cast<const int *>(in.ptr()) + x);

                                    // Quantize
                                    in_value = ((in_value + bias_value + _output_stage->gemmlowp_offset) *
                                                _output_stage->gemmlowp_multiplier) >>
                                                                                    _output_stage->gemmlowp_shift;

                                    // Store the result
                                    *(out.ptr() + x) = static_cast<T>(misc::utility::clamp<int>(in_value, clamp_min,
                                                                                                clamp_max));
                                }
                            },
                            in, bias_i, out);
                } else {
                    execute_window_loop(
                            win,
                            [&](const BICoordinates &) {
                                // Compute 16 elements per iteration
                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                    int32x4x4_t in_s32 = {
                                            {vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 0),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 4),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 8),
                                             vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 12)}};

                                    // Add the offset terms to GEMM's result and multiply by result_mult_int
                                    scale_input(in_s32, result_offset_s32, _output_stage->gemmlowp_multiplier);

                                    wrapper::vstore(reinterpret_cast<T *>(out.ptr() + x),
                                                    finalize_quantization<T>(in_s32, result_shift_s32, min, max));
                                }

                                // Compute left-over elements
                                for (; x < window_end_x; ++x) {
                                    int in_value = *(reinterpret_cast<const int *>(in.ptr()) + x);

                                    // Quantize
                                    in_value = ((in_value + _output_stage->gemmlowp_offset) *
                                                _output_stage->gemmlowp_multiplier) >>
                                                                                    _output_stage->gemmlowp_shift;

                                    // Store the result
                                    *(out.ptr() + x) = static_cast<T>(misc::utility::clamp<int>(in_value, clamp_min,
                                                                                                clamp_max));
                                }
                            },
                            in, out);
                }
            }

            void BICpuGemmLowpQuantizeDownInt32ScaleKernel::configure(BIITensorInfo *src,
                                                                      BIITensorInfo *bias,
                                                                      BIITensorInfo *dst,
                                                                      const BIGEMMLowpOutputStageInfo *output_stage) {
                BI_COMPUTE_UNUSED(bias);
                // Perform validate step
                BI_COMPUTE_ERROR_ON_NULLPTR(src, dst, output_stage);

                // Output auto inizialitation if not yet initialized
                auto_init_if_empty(*dst, src->clone()->set_data_type(output_stage->output_data_type));

                BI_COMPUTE_ERROR_THROW_ON(validate_arguments(src, bias, dst, output_stage));

                _output_stage = output_stage;

                // Configure kernel window
                BIWindow win = calculate_max_window(*src, BISteps());

                BIICpuKernel::configure(win);

                // Check if we need to clamp the result using min and max
                _is_bounded_relu =
                        ((_output_stage->gemmlowp_min_bound != _output_stage->gemmlowp_max_bound) &&
                         !(_output_stage->gemmlowp_min_bound ==
                           std::get<0>(quantization::get_min_max_values_from_quantized_data_type(
                                   output_stage->output_data_type)) &&
                           _output_stage->gemmlowp_max_bound ==
                           std::get<1>(quantization::get_min_max_values_from_quantized_data_type(
                                   output_stage->output_data_type))));
                if (_output_stage->output_data_type == BIDataType::QASYMM8) {
                    _func = &BICpuGemmLowpQuantizeDownInt32ScaleKernel::run_internal<uint8_t>;
                } else if (_output_stage->output_data_type == BIDataType::QASYMM8_SIGNED) {
                    _func = &BICpuGemmLowpQuantizeDownInt32ScaleKernel::run_internal<int8_t>;
                } else {
                    BI_COMPUTE_ERROR("Data type not supported");
                }
            }

            BIStatus BICpuGemmLowpQuantizeDownInt32ScaleKernel::validate(const BIITensorInfo *src,
                                                                         const BIITensorInfo *bias,
                                                                         const BIITensorInfo *dst,
                                                                         const BIGEMMLowpOutputStageInfo *output_stage) {
                BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, bias, dst, output_stage));
                return BIStatus{};
            }

            void BICpuGemmLowpQuantizeDownInt32ScaleKernel::run_op(BIITensorPack &tensors, const BIWindow &window,
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

            const char *BICpuGemmLowpQuantizeDownInt32ScaleKernel::name() const {
                return "BICpuGemmLowpQuantizeDownInt32ScaleKernel";
            }
        } // namespace kernels
    } // namespace cpu
} // namespace BatmanInfer