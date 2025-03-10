//
// Created by Mason on 2025/1/20.
//

#include <cpu/kernels/bi_cpu_gemm_lowp_offset_contribution_output_stage_kernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>
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
                inline int32x4x4_t load_results_input(const BIIterator &mm_result_it, int32_t x) {
                    return {{vld1q_s32(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x + 0),
                             vld1q_s32(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x + 4),
                             vld1q_s32(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x + 8),
                             vld1q_s32(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x + 12)}};
                }

                inline int32x4x4_t load(const int32_t *ptr, int32_t x) {
                    return {{vld1q_s32(ptr + x + 0), vld1q_s32(ptr + x + 4), vld1q_s32(ptr + x + 8),
                             vld1q_s32(ptr + x + 12)}};
                }

                inline int32x4x4_t add_s32(int32x4x4_t a, int32x4_t b) {
                    return {{vaddq_s32(a.val[0], b), vaddq_s32(a.val[1], b), vaddq_s32(a.val[2], b),
                             vaddq_s32(a.val[3], b)}};
                }

                inline int32x4x4_t add_s32(int32x4x4_t a, int32x4x4_t b) {
                    return {{vaddq_s32(a.val[0], b.val[0]), vaddq_s32(a.val[1], b.val[1]),
                             vaddq_s32(a.val[2], b.val[2]),
                             vaddq_s32(a.val[3], b.val[3])}};
                }

                inline int32x4x4_t mul_s32(int32x4x4_t &a, int32_t mul_scalar) {
                    return {{vmulq_n_s32(a.val[0], mul_scalar), vmulq_n_s32(a.val[1], mul_scalar),
                             vmulq_n_s32(a.val[2], mul_scalar),
                             vmulq_n_s32(a.val[3], mul_scalar)}};
                }

                inline int32x4x4_t mul_s32(int32x4x4_t &a, const int32_t *multilpier) {
                    return {{vmulq_s32(a.val[0], vld1q_s32(multilpier)), vmulq_s32(a.val[1], vld1q_s32(multilpier + 4)),
                             vmulq_s32(a.val[2], vld1q_s32(multilpier + 8)),
                             vmulq_s32(a.val[3], vld1q_s32(multilpier + 12))}};
                }

                inline int32x4x4_t get_a_offset(const int32_t *vector_sum_col_ptr, int32_t a_offset, int32_t x) {
                    int32x4x4_t a_offset_term_s32 = load(vector_sum_col_ptr, x);

                    a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], a_offset);
                    a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], a_offset);
                    a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], a_offset);
                    a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], a_offset);
                    return a_offset_term_s32;
                }

                inline int32x4_t get_b_offset(const int32_t *vector_sum_row_ptr, int32_t b_offset) {
                    int32x4_t b_offset_term_s32 = vld1q_dup_s32(vector_sum_row_ptr);
                    b_offset_term_s32 = vmulq_n_s32(b_offset_term_s32, b_offset);
                    return b_offset_term_s32;
                }

                inline int32x4x4_t get_k_offset(int32_t k_offset) {
                    return {{vdupq_n_s32(k_offset), vdupq_n_s32(k_offset), vdupq_n_s32(k_offset),
                             vdupq_n_s32(k_offset)}};
                }

                inline uint8x16_t finalize_quantization_floating_point(
                        int32x4x4_t &in_s32, int32x4_t result_shift_s32, uint8x16_t min_u8, uint8x16_t max_u8,
                        bool is_bounded_relu) {
                    const static int32x4_t zero_s32 = vdupq_n_s32(0);

                    // Shift final result (negative value shift right)
                    in_s32.val[0] = vshlq_s32(in_s32.val[0], result_shift_s32);
                    in_s32.val[1] = vshlq_s32(in_s32.val[1], result_shift_s32);
                    in_s32.val[2] = vshlq_s32(in_s32.val[2], result_shift_s32);
                    in_s32.val[3] = vshlq_s32(in_s32.val[3], result_shift_s32);

                    // Saturate negative values
                    in_s32.val[0] = vmaxq_s32(in_s32.val[0], zero_s32);
                    in_s32.val[1] = vmaxq_s32(in_s32.val[1], zero_s32);
                    in_s32.val[2] = vmaxq_s32(in_s32.val[2], zero_s32);
                    in_s32.val[3] = vmaxq_s32(in_s32.val[3], zero_s32);

                    // Convert S32 to S16
                    const int16x8x2_t in_s16 = {{vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                                                 vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))}};

                    // Convert S16 to U8
                    uint8x16_t out_u8 = vcombine_u8(vqmovun_s16(in_s16.val[0]), vqmovun_s16(in_s16.val[1]));

                    if (is_bounded_relu) {
                        out_u8 = vmaxq_u8(out_u8, min_u8);
                        out_u8 = vminq_u8(out_u8, max_u8);
                    }

                    return out_u8;
                }

                inline int8x16_t finalize_quantization_floating_point(
                        int32x4x4_t &in_s32, int32x4_t result_shift_s32, int8x16_t min_s8, int8x16_t max_s8,
                        bool is_bounded_relu) {
                    const static int32x4_t zero_s32 = vdupq_n_s32(0);

                    // Shift final result (negative value shift right)
                    in_s32.val[0] = vshlq_s32(in_s32.val[0], result_shift_s32);
                    in_s32.val[1] = vshlq_s32(in_s32.val[1], result_shift_s32);
                    in_s32.val[2] = vshlq_s32(in_s32.val[2], result_shift_s32);
                    in_s32.val[3] = vshlq_s32(in_s32.val[3], result_shift_s32);

                    // Saturate negative values
                    in_s32.val[0] = vmaxq_s32(in_s32.val[0], zero_s32);
                    in_s32.val[1] = vmaxq_s32(in_s32.val[1], zero_s32);
                    in_s32.val[2] = vmaxq_s32(in_s32.val[2], zero_s32);
                    in_s32.val[3] = vmaxq_s32(in_s32.val[3], zero_s32);

                    // Convert S32 to S16
                    const int16x8x2_t in_s16 = {{vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                                                 vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))}};

                    // Convert S16 to S8
                    int8x16_t out_s8 = vcombine_s8(vqmovn_s16(in_s16.val[0]), vqmovn_s16(in_s16.val[1]));

                    if (is_bounded_relu) {
                        out_s8 = vmaxq_s8(out_s8, min_s8);
                        out_s8 = vminq_s8(out_s8, max_s8);
                    }

                    return out_s8;
                }

                inline int8x16_t finalize_quantization_floating_point(
                        int32x4x4_t &in_s32, int32x4x4_t result_shift_s32, int8x16_t min_s8, int8x16_t max_s8,
                        bool is_bounded_relu) {
                    const static int32x4_t zero_s32 = vdupq_n_s32(0);

                    // Shift final result (negative value shift right)
                    in_s32.val[0] = vshlq_s32(in_s32.val[0], vnegq_s32(result_shift_s32.val[0]));
                    in_s32.val[1] = vshlq_s32(in_s32.val[1], vnegq_s32(result_shift_s32.val[1]));
                    in_s32.val[2] = vshlq_s32(in_s32.val[2], vnegq_s32(result_shift_s32.val[2]));
                    in_s32.val[3] = vshlq_s32(in_s32.val[3], vnegq_s32(result_shift_s32.val[3]));

                    // Saturate negative values
                    in_s32.val[0] = vmaxq_s32(in_s32.val[0], zero_s32);
                    in_s32.val[1] = vmaxq_s32(in_s32.val[1], zero_s32);
                    in_s32.val[2] = vmaxq_s32(in_s32.val[2], zero_s32);
                    in_s32.val[3] = vmaxq_s32(in_s32.val[3], zero_s32);

                    // Convert S32 to S16
                    const int16x8x2_t in_s16 = {{vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                                                 vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))}};

                    // Convert S16 to S8
                    int8x16_t out_s8 = vcombine_s8(vqmovn_s16(in_s16.val[0]), vqmovn_s16(in_s16.val[1]));

                    if (is_bounded_relu) {
                        out_s8 = vmaxq_s8(out_s8, min_s8);
                        out_s8 = vminq_s8(out_s8, max_s8);
                    }

                    return out_s8;
                }

                template<typename T>
                struct VectorTyper {
                    using stype = T;
                    using vtype = typename wrapper::traits::neon_bitvector_t<T, wrapper::traits::BitWidth::W128>;
                };

                inline BIWindow get_win_vector_sum(const BIWindow &window) {
                    BIWindow win_vector_sum(window);
                    win_vector_sum.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                    win_vector_sum.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));
                    return win_vector_sum;
                }

                inline BIIterator get_vector_sum_col_it(const BIWindow &window, const BIITensor *vector_sum_col) {
                    BIIterator vector_sum_col_it(vector_sum_col, get_win_vector_sum(window));
                    return vector_sum_col_it;
                }

                inline BIIterator get_vector_sum_row_it(const BIWindow &window, const BIITensor *vector_sum_row) {
                    BIWindow win_vector_sum_row = get_win_vector_sum(window);
                    win_vector_sum_row.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                    BIIterator vector_sum_row_it(vector_sum_row, win_vector_sum_row);
                    return vector_sum_row_it;
                }

                inline BIIterator get_bias_it(const BIWindow &window, const BIITensor *bias) {
                    BIWindow win_bias(window);
                    win_bias.set(BIWindow::DimY, BIWindow::BIDimension(0, 1, 1));
                    win_bias.set(BIWindow::DimZ, BIWindow::BIDimension(0, 1, 1));
                    BIIterator bias_it(bias, win_bias);
                    return bias_it;
                }

                template<typename VT>
                inline void run_offset_contribution_output_stage_window(const int32_t *vector_sum_col_ptr,
                                                                        const int32_t *vector_sum_row_ptr,
                                                                        const int32_t *bias_ptr,
                                                                        BIIterator mm_result_it,
                                                                        BIIterator out_it,
                                                                        const int32x4_t result_offset_s32,
                                                                        const int32x4_t result_shift_s32,
                                                                        typename VT::vtype min_vec,
                                                                        typename VT::vtype max_vec,
                                                                        int32_t a_offset,
                                                                        int32_t b_offset,
                                                                        int32_t k_offset,
                                                                        int32_t multiplier,
                                                                        int32_t shift,
                                                                        int32_t offset,
                                                                        int32_t min_bound,
                                                                        int32_t max_bound,
                                                                        int window_step_x,
                                                                        int window_start_x,
                                                                        int window_end_x,
                                                                        bool has_a_offset,
                                                                        bool has_b_offset,
                                                                        bool has_bias,
                                                                        bool is_bounded_relu,
                                                                        bool is_fixed_point) {
                    int32x4x4_t offset_term_s32 = {0, 0, 0, 0};
                    if (!is_fixed_point) {
                        // Combine quantization offset with other offsets.
                        offset_term_s32 = add_s32(offset_term_s32, result_offset_s32);
                    }
                    if (has_a_offset && has_b_offset) {
                        offset_term_s32 = add_s32(offset_term_s32, get_k_offset(k_offset));
                    }
                    if (has_b_offset) {
                        offset_term_s32 = add_s32(offset_term_s32, get_b_offset(vector_sum_row_ptr, b_offset));
                    }

                    int x = window_start_x;
                    for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                        int32x4x4_t in_s32 = load_results_input(mm_result_it, x);

                        if (has_a_offset) {
                            in_s32 = add_s32(in_s32, get_a_offset(vector_sum_col_ptr, a_offset, x));
                        }
                        if (has_bias) {
                            in_s32 = add_s32(in_s32, load(bias_ptr, x));
                        }
                        if (!is_fixed_point || has_b_offset) {
                            in_s32 = add_s32(in_s32, offset_term_s32);
                        }
                        if (!is_fixed_point) {
                            in_s32 = mul_s32(in_s32, multiplier);
                        }

                        if (is_fixed_point) {
                            wrapper::vstore(
                                    reinterpret_cast<typename VT::stype *>(out_it.ptr() + x),
                                    finalize_quantization(in_s32, multiplier, shift, result_offset_s32, min_vec,
                                                          max_vec, is_bounded_relu));
                        } else {
                            wrapper::vstore(
                                    reinterpret_cast<typename VT::stype *>(out_it.ptr() + x),
                                    finalize_quantization_floating_point(in_s32, result_shift_s32, min_vec, max_vec,
                                                                         is_bounded_relu));
                        }
                    }
                    // Compute left-over elements
                    for (; x < window_end_x; ++x) {
                        int32_t in_value =
                                *(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x) +
                                wrapper::vgetlane(offset_term_s32.val[0], 0);

                        if (has_a_offset) {
                            in_value += (*(vector_sum_col_ptr + x) * a_offset);
                        }
                        if (has_bias) {
                            in_value += *(bias_ptr + x);
                        }

                        if (is_fixed_point) {
                            // Finalize and store the result
                            *reinterpret_cast<typename VT::stype *>(out_it.ptr() + x) =
                                    finalize_quantization(in_value, multiplier, shift, offset,
                                                          static_cast<typename VT::stype>(min_bound),
                                                          static_cast<typename VT::stype>(max_bound), is_bounded_relu);
                        } else {
                            // Finalize quantization
                            in_value = (in_value * multiplier) >> shift;

                            // Bound and store the result
                            if (is_bounded_relu) {
                                in_value = static_cast<typename VT::stype>(
                                        std::max<int32_t>(min_bound, std::min<int32_t>(max_bound, in_value)));
                            }
                            *reinterpret_cast<typename VT::stype *>(out_it.ptr() + x) =
                                    static_cast<typename VT::stype>(std::max<int32_t>(
                                            static_cast<int32_t>(std::numeric_limits<typename VT::stype>::lowest()),
                                            std::min<int32_t>(
                                                    static_cast<int32_t>(std::numeric_limits<typename VT::stype>::max()),
                                                    in_value)));
                        }
                    }
                }

                inline void run_offset_contribution_output_stage_window_symm(const int32_t *vector_sum_col_ptr,
                                                                             const int32_t *bias_ptr,
                                                                             BIIterator mm_result_it,
                                                                             BIIterator out_it,
                                                                             const int32_t *result_multipliers,
                                                                             const int32_t *result_shifts,
                                                                             const int32x4_t result_offset,
                                                                             int8x16_t min_s8,
                                                                             int8x16_t max_s8,
                                                                             int32_t a_offset,
                                                                             int32_t offset,
                                                                             int32_t min_bound,
                                                                             int32_t max_bound,
                                                                             int window_step_x,
                                                                             int window_start_x,
                                                                             int window_end_x,
                                                                             bool has_a_offset,
                                                                             bool has_bias,
                                                                             bool is_bounded_relu,
                                                                             bool is_fixed_point) {
                    int32x4x4_t offset_term_s32 = {0, 0, 0, 0};
                    if (!is_fixed_point) {
                        // Combine quantization offset with other offsets.
                        offset_term_s32 = add_s32(offset_term_s32, result_offset);
                    }

                    int x = window_start_x;
                    for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                        int32x4x4_t in_s32 = load_results_input(mm_result_it, x);

                        if (has_a_offset) {
                            in_s32 = add_s32(in_s32, get_a_offset(vector_sum_col_ptr, a_offset, x));
                        }
                        if (has_bias) {
                            in_s32 = add_s32(in_s32, load(bias_ptr, x));
                        }
                        if (!is_fixed_point) {
                            in_s32 = add_s32(in_s32, offset_term_s32);
                            in_s32 = mul_s32(in_s32, result_multipliers + x);
                        }

                        if (is_fixed_point) {
                            vst1q_s8(reinterpret_cast<int8_t *>(out_it.ptr() + x),
                                     finalize_quantization_symm(in_s32, load(result_multipliers, x),
                                                                load(result_shifts, x),
                                                                result_offset, min_s8, max_s8, is_bounded_relu));
                        } else {
                            vst1q_s8(
                                    reinterpret_cast<int8_t *>(out_it.ptr() + x),
                                    finalize_quantization_floating_point(in_s32, load(result_shifts, x), min_s8, max_s8,
                                                                         is_bounded_relu));
                        }
                    }
                    // Compute left-over elements
                    for (; x < window_end_x; ++x) {
                        int32_t in_value =
                                *(reinterpret_cast<const int32_t *>(mm_result_it.ptr()) + x) +
                                wrapper::vgetlane(offset_term_s32.val[0], 0);

                        if (has_a_offset) {
                            in_value += (*(vector_sum_col_ptr + x) * a_offset);
                        }
                        if (has_bias) {
                            in_value += *(bias_ptr + x);
                        }

                        if (is_fixed_point) {
                            // Finalize and store the result
                            *(out_it.ptr() + x) =
                                    finalize_quantization(in_value, result_multipliers[x], result_shifts[x], offset,
                                                          static_cast<int8_t>(min_bound),
                                                          static_cast<int8_t>(max_bound), is_bounded_relu);
                        } else {
                            // Finalize quantization
                            in_value = (in_value * result_multipliers[x]) >> (-result_shifts[x]);

                            // Bound and store the result
                            if (is_bounded_relu) {
                                in_value = static_cast<int8_t>(std::max<int32_t>(min_bound, std::min<int32_t>(max_bound,
                                                                                                              in_value)));
                            }
                            *(out_it.ptr() + x) = static_cast<int8_t>(std::max<int32_t>(-128, std::min<int32_t>(127,
                                                                                                                in_value)));
                        }
                    }
                }

                template<typename T>
                void run_offset_contribution_output_stage(const BIWindow &window,
                                                          const BIITensor *mm_result,
                                                          const BIITensor *vector_sum_col,
                                                          const BIITensor *vector_sum_row,
                                                          const BIITensor *bias,
                                                          BIITensor *output,
                                                          int32_t a_offset,
                                                          int32_t b_offset,
                                                          int32_t k_offset,
                                                          bool is_vector_sum_col_batched,
                                                          BIGEMMLowpOutputStageInfo output_stage,
                                                          bool is_gemm3d,
                                                          bool is_bounded_relu,
                                                          bool is_fixed_point) {
                    //  Semantics of XYZW Explained for each tensor
                    //
                    //  | Tensor            |    XYZW when is_gemm3d == false       |    XYZW when is_gemm3d == true                    |
                    // -------------------------------------------------------------------------------------------------------------------
                    //  | mm_result         |  x -> width,  y -> height, z -> batch |  x -> width, y -> height, z -> depth, w -> batch  |
                    //  | collapsed window  |  x -> width,  y -> height, z -> batch |  x -> width, y -> height, z -> depth * batch      |
                    //  | vector_sum_row    |  x -> height, y -> batch              |  x -> height * depth, y -> batch                  |
                    //  | Vector_sum_col    |  x -> width,  y -> batch              |  x -> width, y -> batch                           |

                    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;
                    using Typer = VectorTyper<T>;

                    const int height_input = is_gemm3d ? mm_result->info()->dimension(1) : 0;
                    const int depth_input = is_gemm3d ? mm_result->info()->dimension(2) : 1;

                    const int32_t multiplier = output_stage.gemmlowp_multiplier;
                    const int32_t shift = output_stage.gemmlowp_shift;
                    const int32_t offset = output_stage.gemmlowp_offset;
                    const int32_t min_bound = output_stage.gemmlowp_min_bound;
                    const int32_t max_bound = output_stage.gemmlowp_max_bound;

                    const int32x4_t result_offset_s32 = vdupq_n_s32(offset);
                    const int32x4_t result_shift_s32 = vdupq_n_s32(is_fixed_point ? shift : -shift);
                    const auto min_vec = wrapper::vdup_n(static_cast<T>(min_bound), ExactTagType{});
                    const auto max_vec = wrapper::vdup_n(static_cast<T>(max_bound), ExactTagType{});

                    const int window_step_x = 16;
                    const auto window_start_x = static_cast<int>(window.x().start());
                    const auto window_end_x = static_cast<int>(window.x().end());

                    BIWindow win(window);
                    win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                    BIWindow collapsed_window = win.collapse_if_possible(win, BIWindow::DimZ);

                    BIIterator mm_result_it(mm_result, win);
                    BIIterator out_it(output, win);

                    if ((a_offset != 0) && (b_offset != 0)) {
                        BI_COMPUTE_ERROR_ON_NULLPTR(vector_sum_col);
                        BI_COMPUTE_ERROR_ON_NULLPTR(vector_sum_row);

                        BIIterator vector_sum_col_it = get_vector_sum_col_it(collapsed_window, vector_sum_col);
                        BIIterator vector_sum_row_it = get_vector_sum_row_it(collapsed_window, vector_sum_row);

                        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

                        // Offset in case vector_sum_col is batched in y dimension
                        const int vector_sum_col_stride_batch =
                                is_vector_sum_col_batched ? vector_sum_col->info()->strides_in_bytes().y() : 0;

                        if (bias != nullptr) {
                            BIIterator bias_it = get_bias_it(collapsed_window, bias);
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &id) {
                                        const int batch_id = id.z() / depth_input;
                                        const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                                vector_sum_col_it.ptr() + batch_id * vector_sum_col_stride_batch);
                                        const auto vector_sum_row_ptr =
                                                reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() +
                                                                                  batch_id * sum_row_stride_y) +
                                                id.y() + (id.z() % depth_input) * height_input;
                                        run_offset_contribution_output_stage_window<Typer>(
                                                vector_sum_col_ptr, vector_sum_row_ptr,
                                                reinterpret_cast<const int32_t *>(bias_it.ptr()),
                                                mm_result_it, out_it, result_offset_s32, result_shift_s32, min_vec,
                                                max_vec, a_offset, b_offset,
                                                k_offset, multiplier, shift, offset, min_bound, max_bound,
                                                window_step_x, window_start_x,
                                                window_end_x, true, true, true, is_bounded_relu, is_fixed_point);
                                    },
                                    vector_sum_col_it, vector_sum_row_it, bias_it, mm_result_it, out_it);
                        } else {
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &id) {
                                        const int batch_id = id.z() / depth_input;
                                        const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                                vector_sum_col_it.ptr() + batch_id * vector_sum_col_stride_batch);
                                        const auto vector_sum_row_ptr =
                                                reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() +
                                                                                  batch_id * sum_row_stride_y) +
                                                id.y() + (id.z() % depth_input) * height_input;
                                        run_offset_contribution_output_stage_window<Typer>(
                                                vector_sum_col_ptr, vector_sum_row_ptr, nullptr, mm_result_it, out_it,
                                                result_offset_s32,
                                                result_shift_s32, min_vec, max_vec, a_offset, b_offset, k_offset,
                                                multiplier, shift, offset,
                                                min_bound, max_bound, window_step_x, window_start_x, window_end_x, true,
                                                true, false,
                                                is_bounded_relu, is_fixed_point);
                                    },
                                    vector_sum_col_it, vector_sum_row_it, mm_result_it, out_it);
                        }
                    } else if ((a_offset == 0) && (b_offset != 0)) {
                        BI_COMPUTE_ERROR_ON_NULLPTR(vector_sum_row);

                        BIIterator vector_sum_row_it = get_vector_sum_row_it(collapsed_window, vector_sum_row);

                        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

                        if (bias != nullptr) {
                            BIIterator bias_it = get_bias_it(collapsed_window, bias);
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &id) {
                                        const int batch_id = id.z() / depth_input;
                                        const auto vector_sum_row_ptr =
                                                reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() +
                                                                                  batch_id * sum_row_stride_y) +
                                                id.y() + (id.z() % depth_input) * height_input;
                                        run_offset_contribution_output_stage_window<Typer>(
                                                nullptr, vector_sum_row_ptr,
                                                reinterpret_cast<const int32_t *>(bias_it.ptr()), mm_result_it,
                                                out_it, result_offset_s32, result_shift_s32, min_vec, max_vec, a_offset,
                                                b_offset, k_offset,
                                                multiplier, shift, offset, min_bound, max_bound, window_step_x,
                                                window_start_x, window_end_x,
                                                false, true, true, is_bounded_relu, is_fixed_point);
                                    },
                                    vector_sum_row_it, bias_it, mm_result_it, out_it);
                        } else {
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &id) {
                                        const int batch_id = id.z() / depth_input;
                                        const auto vector_sum_row_ptr =
                                                reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() +
                                                                                  batch_id * sum_row_stride_y) +
                                                id.y() + (id.z() % depth_input) * height_input;
                                        run_offset_contribution_output_stage_window<Typer>(
                                                nullptr, vector_sum_row_ptr, nullptr, mm_result_it, out_it,
                                                result_offset_s32, result_shift_s32,
                                                min_vec, max_vec, a_offset, b_offset, k_offset, multiplier, shift,
                                                offset, min_bound, max_bound,
                                                window_step_x, window_start_x, window_end_x, false, true, false,
                                                is_bounded_relu,
                                                is_fixed_point);
                                    },
                                    vector_sum_row_it, mm_result_it, out_it);
                        }
                    } else if ((a_offset != 0) && (b_offset == 0)) {
                        BI_COMPUTE_ERROR_ON_NULLPTR(vector_sum_col);

                        BIIterator vector_sum_col_it = get_vector_sum_col_it(collapsed_window, vector_sum_col);

                        // Offset in case vector_sum_col is batched in y dimension
                        const int vector_sum_col_stride_batch =
                                is_vector_sum_col_batched ? vector_sum_col->info()->strides_in_bytes().y() : 0;

                        if (bias != nullptr) {
                            BIIterator bias_it = get_bias_it(collapsed_window, bias);
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &id) {
                                        const int batch_id = id.z() / depth_input;
                                        const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                                vector_sum_col_it.ptr() + batch_id * vector_sum_col_stride_batch);
                                        run_offset_contribution_output_stage_window<Typer>(
                                                vector_sum_col_ptr, nullptr,
                                                reinterpret_cast<const int32_t *>(bias_it.ptr()), mm_result_it,
                                                out_it, result_offset_s32, result_shift_s32, min_vec, max_vec, a_offset,
                                                b_offset, k_offset,
                                                multiplier, shift, offset, min_bound, max_bound, window_step_x,
                                                window_start_x, window_end_x,
                                                true, false, true, is_bounded_relu, is_fixed_point);
                                    },
                                    vector_sum_col_it, bias_it, mm_result_it, out_it);
                        } else {
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &id) {
                                        const int batch_id = id.z() / depth_input;
                                        const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                                vector_sum_col_it.ptr() + batch_id * vector_sum_col_stride_batch);
                                        run_offset_contribution_output_stage_window<Typer>(
                                                vector_sum_col_ptr, nullptr, nullptr, mm_result_it, out_it,
                                                result_offset_s32, result_shift_s32,
                                                min_vec, max_vec, a_offset, b_offset, k_offset, multiplier, shift,
                                                offset, min_bound, max_bound,
                                                window_step_x, window_start_x, window_end_x, true, false, false,
                                                is_bounded_relu,
                                                is_fixed_point);
                                    },
                                    vector_sum_col_it, mm_result_it, out_it);
                        }
                    } else {
                        if (bias != nullptr) {
                            BIIterator bias_it = get_bias_it(collapsed_window, bias);
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &) {
                                        run_offset_contribution_output_stage_window<Typer>(
                                                nullptr, nullptr, reinterpret_cast<const int32_t *>(bias_it.ptr()),
                                                mm_result_it, out_it,
                                                result_offset_s32, result_shift_s32, min_vec, max_vec, a_offset,
                                                b_offset, k_offset, multiplier,
                                                shift, offset, min_bound, max_bound, window_step_x, window_start_x,
                                                window_end_x, false, false,
                                                true, is_bounded_relu, is_fixed_point);
                                    },
                                    bias_it, mm_result_it, out_it);
                        } else {
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &) {
                                        run_offset_contribution_output_stage_window<Typer>(
                                                nullptr, nullptr, nullptr, mm_result_it, out_it, result_offset_s32,
                                                result_shift_s32, min_vec,
                                                max_vec, a_offset, b_offset, k_offset, multiplier, shift, offset,
                                                min_bound, max_bound,
                                                window_step_x, window_start_x, window_end_x, false, false, false,
                                                is_bounded_relu,
                                                is_fixed_point);
                                    },
                                    mm_result_it, out_it);
                        }
                        return;
                    }
                }

                void run_offset_contribution_output_stage_symm(const BIWindow &window,
                                                               const BIITensor *mm_result,
                                                               const BIITensor *vector_sum_col,
                                                               const BIITensor *vector_sum_row,
                                                               const BIITensor *bias,
                                                               BIITensor *output,
                                                               int32_t a_offset,
                                                               int32_t b_offset,
                                                               int32_t k_offset,
                                                               bool is_vector_sum_col_batched,
                                                               BIGEMMLowpOutputStageInfo output_stage,
                                                               bool is_gemm3d,
                                                               bool is_bounded_relu,
                                                               bool is_fixed_point) {
                    BI_COMPUTE_UNUSED(vector_sum_row, b_offset, k_offset);

                    const int depth_input = is_gemm3d ? mm_result->info()->dimension(2) : 1;

                    const int32_t offset = output_stage.gemmlowp_offset;
                    const int32_t min_bound = output_stage.gemmlowp_min_bound;
                    const int32_t max_bound = output_stage.gemmlowp_max_bound;

                    const int32_t *result_multipliers = output_stage.gemmlowp_multipliers.data();
                    const int32_t *result_shifts = output_stage.gemmlowp_shifts.data();
                    const int32x4_t result_offset_s32 = vdupq_n_s32(offset);
                    const int8x16_t min_s8 = vdupq_n_s8(static_cast<int8_t>(min_bound));
                    const int8x16_t max_s8 = vdupq_n_s8(static_cast<int8_t>(max_bound));

                    const int window_step_x = 16;
                    const auto window_start_x = static_cast<int>(window.x().start());
                    const auto window_end_x = static_cast<int>(window.x().end());

                    BIWindow win(window);
                    win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                    BIWindow collapsed_window = win.collapse_if_possible(win, BIWindow::DimZ);

                    BIIterator mm_result_it(mm_result, win);
                    BIIterator out_it(output, win);

                    if (a_offset != 0) {
                        BI_COMPUTE_ERROR_ON_NULLPTR(vector_sum_col);

                        BIIterator vector_sum_col_it = get_vector_sum_col_it(collapsed_window, vector_sum_col);

                        // Offset in case vector_sum_col is batched in y dimension
                        const int vector_sum_col_stride_batch =
                                is_vector_sum_col_batched ? vector_sum_col->info()->strides_in_bytes().y() : 0;

                        if (bias != nullptr) {
                            BIIterator bias_it = get_bias_it(collapsed_window, bias);
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &id) {
                                        const int batch_id = id.z() / depth_input;
                                        const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                                vector_sum_col_it.ptr() + batch_id * vector_sum_col_stride_batch);
                                        run_offset_contribution_output_stage_window_symm(
                                                vector_sum_col_ptr, reinterpret_cast<const int32_t *>(bias_it.ptr()),
                                                mm_result_it, out_it,
                                                result_multipliers, result_shifts, result_offset_s32, min_s8, max_s8,
                                                a_offset, offset,
                                                min_bound, max_bound, window_step_x, window_start_x, window_end_x, true,
                                                true, is_bounded_relu,
                                                is_fixed_point);
                                    },
                                    vector_sum_col_it, bias_it, mm_result_it, out_it);
                        } else {
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &id) {
                                        const int batch_id = id.z() / depth_input;
                                        const auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                                vector_sum_col_it.ptr() + batch_id * vector_sum_col_stride_batch);
                                        run_offset_contribution_output_stage_window_symm(
                                                vector_sum_col_ptr, nullptr, mm_result_it, out_it, result_multipliers,
                                                result_shifts,
                                                result_offset_s32, min_s8, max_s8, a_offset, offset, min_bound,
                                                max_bound, window_step_x,
                                                window_start_x, window_end_x, true, false, is_bounded_relu,
                                                is_fixed_point);
                                    },
                                    vector_sum_col_it, mm_result_it, out_it);
                        }
                    } else {
                        if (bias != nullptr) {
                            BIIterator bias_it = get_bias_it(collapsed_window, bias);
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &) {
                                        run_offset_contribution_output_stage_window_symm(
                                                nullptr, reinterpret_cast<const int32_t *>(bias_it.ptr()), mm_result_it,
                                                out_it,
                                                result_multipliers, result_shifts, result_offset_s32, min_s8, max_s8,
                                                a_offset, offset,
                                                min_bound, max_bound, window_step_x, window_start_x, window_end_x,
                                                false, true, is_bounded_relu,
                                                is_fixed_point);
                                    },
                                    bias_it, mm_result_it, out_it);
                        } else {
                            execute_window_loop(
                                    collapsed_window,
                                    [&](const BICoordinates &) {
                                        run_offset_contribution_output_stage_window_symm(
                                                nullptr, nullptr, mm_result_it, out_it, result_multipliers,
                                                result_shifts, result_offset_s32,
                                                min_s8, max_s8, a_offset, offset, min_bound, max_bound, window_step_x,
                                                window_start_x,
                                                window_end_x, false, false, is_bounded_relu, is_fixed_point);
                                    },
                                    mm_result_it, out_it);
                        }
                        return;
                    }
                }

                BIStatus validate_arguments(const BIITensorInfo *mm_result,
                                            const BIITensorInfo *vector_sum_col,
                                            const BIITensorInfo *vector_sum_row,
                                            const BIITensorInfo *bias,
                                            const BIITensorInfo *output,
                                            int32_t a_offset,
                                            int32_t b_offset,
                                            BIGEMMLowpOutputStageInfo output_stage) {
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mm_result, 1, BIDataType::S32);
                    if (output->data_type() != BIDataType::QASYMM8) {
                        BI_COMPUTE_RETURN_ERROR_ON(
                                mm_result->dimension(0) > 1 && output_stage.gemmlowp_multipliers.size() > 1 &&
                                b_offset != 0);
                    }
                    BI_COMPUTE_RETURN_ERROR_ON(output_stage.gemmlowp_min_bound > output_stage.gemmlowp_max_bound);
                    BI_COMPUTE_RETURN_ERROR_ON(output_stage.type != BIGEMMLowpOutputStageType::QUANTIZE_DOWN &&
                                               output_stage.type !=
                                               BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT);

                    if (bias != nullptr) {
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, BIDataType::S32);
                        BI_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
                        BI_COMPUTE_RETURN_ERROR_ON(mm_result->dimension(0) != bias->dimension(0));
                    }

                    // If a_offset == 0, vector_sum_col can be a nullptr
                    if (a_offset != 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_col, 1, BIDataType::S32);
                        BI_COMPUTE_RETURN_ERROR_ON(vector_sum_col->dimension(0) != mm_result->dimension(0));
                        BI_COMPUTE_RETURN_ERROR_ON(vector_sum_col->num_dimensions() > 2);
                    }

                    // If b_offset == 0, vector_sum_row can be a nullptr
                    if (b_offset != 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_row, 1, BIDataType::S32);

                        // Check if input is a 3D reinterpretation
                        const bool reinterpret_as_3d =
                                mm_result->num_dimensions() > 1 &&
                                mm_result->tensor_shape().y() != vector_sum_row->tensor_shape().x();

                        // Validate input
                        BI_COMPUTE_RETURN_ERROR_ON(reinterpret_as_3d && vector_sum_row->dimension(0) !=
                                                                        (mm_result->dimension(1) *
                                                                         mm_result->dimension(2)));
                        BI_COMPUTE_RETURN_ERROR_ON(
                                !reinterpret_as_3d && vector_sum_row->dimension(0) != mm_result->dimension(1));

                        BITensorShape output_shape = output->tensor_shape();
                        if (output_shape.num_dimensions() > 1) {
                            const unsigned int output_batch_idx = reinterpret_as_3d ? 3 : 2;

                            BITensorShape vector_sum_row_shape = vector_sum_row->tensor_shape();
                            vector_sum_row_shape.collapse_from(1);
                            output_shape.collapse_from(output_batch_idx);

                            BI_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_row_shape[1] != output_shape[output_batch_idx],
                                                           "mm_result tensor must have the same number of batches of output tensor");

                            if (a_offset != 0) {
                                BITensorShape vector_sum_col_shape = vector_sum_col->tensor_shape();
                                vector_sum_col_shape.collapse_from(1);

                                BI_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_col_shape[1] != 1 &&
                                                               vector_sum_col_shape[1] != vector_sum_row_shape[1],
                                                               "vector_sum_col tensor must have the same number of batches of "
                                                               "vector_sum_row_shape or the number of batches must be set to 1");
                            }
                        }

                        // Check Tensor Rank of vector_sum_row
                        BI_COMPUTE_RETURN_ERROR_ON(vector_sum_row->num_dimensions() > 3);
                    }

                    if (output->total_size() != 0) {
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, BIDataType::QASYMM8,
                                                                            BIDataType::QASYMM8_SIGNED);
                        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mm_result, output);
                    }

                    return BIStatus{};
                }
            } // namespace

            void BICpuGemmLowpOffsetContributionOutputStageKernel::configure(const BIITensorInfo *mm_result,
                                                                             const BIITensorInfo *vector_sum_col,
                                                                             const BIITensorInfo *vector_sum_row,
                                                                             const BIITensorInfo *bias,
                                                                             BIITensorInfo *dst,
                                                                             int32_t k,
                                                                             int32_t a_offset,
                                                                             int32_t b_offset,
                                                                             BIGEMMLowpOutputStageInfo output_stage) {
                BI_COMPUTE_UNUSED(vector_sum_row, bias);
                // Perform validate step
                BI_COMPUTE_ERROR_ON_NULLPTR(mm_result, dst);
                BI_COMPUTE_ERROR_THROW_ON(
                        validate_arguments(mm_result, vector_sum_col, vector_sum_row, bias, dst, a_offset, b_offset,
                                           output_stage));

                _a_offset = a_offset;
                _b_offset = b_offset;
                _k = k;
                _output_stage = output_stage;

                // If a_offset == 0, vector_sum_col can be a nullptr
                if (a_offset != 0) {
                    // Check if vector_sum_col_shape should be slidden or not
                    // Don't slide vector_sum_col_shape along the y dimension if vector_sum_col_shape has just 1 dimension and vector_sum_row_shape more than 1
                    // This scenario can happen when the matrix multiplication is used to perform a convolution operation
                    _is_vector_sum_col_batched = vector_sum_col->tensor_shape().num_dimensions() > 1;
                }

                // Output auto inizialitation if not yet initialized
                auto_init_if_empty(*dst, mm_result->clone()->set_data_type(BIDataType::QASYMM8));

                // Configure kernel window
                BIWindow win = calculate_max_window(*mm_result, BISteps());

                // Note: This kernel performs 16 elements per iteration.
                // However, since we use a left-over for loop, we cannot have any read or write out of memory
                // For this reason num_elems_processed_per_iteration is 1 and so update_window_and_padding() can be skipped
                BIICpuKernel::configure(win);
            }

            BIStatus BICpuGemmLowpOffsetContributionOutputStageKernel::validate(const BIITensorInfo *mm_result,
                                                                                const BIITensorInfo *vector_sum_col,
                                                                                const BIITensorInfo *vector_sum_row,
                                                                                const BIITensorInfo *bias,
                                                                                const BIITensorInfo *output,
                                                                                int32_t a_offset,
                                                                                int32_t b_offset,
                                                                                BIGEMMLowpOutputStageInfo output_stage) {
                BI_COMPUTE_ERROR_ON_NULLPTR(mm_result, output);
                BI_COMPUTE_RETURN_ON_ERROR(
                        validate_arguments(mm_result, vector_sum_col, vector_sum_row, bias, output, a_offset, b_offset,
                                           output_stage));
                return BIStatus{};
            }

            void BICpuGemmLowpOffsetContributionOutputStageKernel::set_a_offset(int32_t a_offset) {
                _a_offset = a_offset;
            }

            void BICpuGemmLowpOffsetContributionOutputStageKernel::set_b_offset(int32_t b_offset) {
                _b_offset = b_offset;
            }

            void BICpuGemmLowpOffsetContributionOutputStageKernel::run_op(BIITensorPack &tensors,
                                                                          const BIWindow &window,
                                                                          const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                auto mm_result = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
                auto vector_sum_col = tensors.get_const_tensor(BITensorType::ACL_SRC_1);
                auto vector_sum_row = tensors.get_const_tensor(BITensorType::ACL_SRC_2);
                auto bias = tensors.get_const_tensor(BITensorType::ACL_SRC_3);
                auto dst = tensors.get_tensor(BITensorType::ACL_DST);

                BIPixelValue type_min{};
                BIPixelValue type_max{};
                std::tie(type_min, type_max) = get_min_max(dst->info()->data_type());
                int32_t type_min_int = type_min.get<int32_t>();
                int32_t type_max_int = type_max.get<int32_t>();

                const bool reinterpret_as_3d = vector_sum_row != nullptr && mm_result->info()->num_dimensions() > 1 &&
                                               mm_result->info()->tensor_shape().y() !=
                                               vector_sum_row->info()->tensor_shape().x();

                const bool is_bounded_relu =
                        !(_output_stage.gemmlowp_min_bound <= type_min_int &&
                          _output_stage.gemmlowp_max_bound >= type_max_int);

                // Check if we need to perform fixed point requantization
                const bool is_fixed_point = _output_stage.type != BIGEMMLowpOutputStageType::QUANTIZE_DOWN;

                // Check if symmetric per-channel execution
                const bool is_signed = dst->info()->data_type() == BIDataType::QASYMM8_SIGNED;

                // Check if symmetric per-channel execution
                const bool is_symm = _output_stage.is_quantized_per_channel;

                auto k_offset = _a_offset * _b_offset * _k;
                if (is_symm) {
                    run_offset_contribution_output_stage_symm(window, mm_result, vector_sum_col, vector_sum_row, bias,
                                                              dst,
                                                              _a_offset, _b_offset, k_offset,
                                                              _is_vector_sum_col_batched,
                                                              _output_stage, reinterpret_as_3d, is_bounded_relu,
                                                              is_fixed_point);
                } else {
                    if (is_signed) {
                        run_offset_contribution_output_stage<int8_t>(
                                window, mm_result, vector_sum_col, vector_sum_row, bias, dst, _a_offset, _b_offset,
                                k_offset,
                                _is_vector_sum_col_batched, _output_stage, reinterpret_as_3d, is_bounded_relu,
                                is_fixed_point);
                    } else {
                        run_offset_contribution_output_stage<uint8_t>(
                                window, mm_result, vector_sum_col, vector_sum_row, bias, dst, _a_offset, _b_offset,
                                k_offset,
                                _is_vector_sum_col_batched, _output_stage, reinterpret_as_3d, is_bounded_relu,
                                is_fixed_point);
                    }
                }
            }

            const char *BICpuGemmLowpOffsetContributionOutputStageKernel::name() const {
                return "BICpuGemmLowpOffsetContributionOutputStageKernel";
            }
        }
    }
}
