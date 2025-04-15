//
// Created by Mason on 2025/1/20.
//

#include <cpu/kernels/bi_cpu_gemm_lowp_offset_contribution_kernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>
#include <data/core/bi_window.hpp>

#include <data/core/helpers/bi_window_helpers.hpp>

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace {
                BIStatus validate_arguments(const BIITensorInfo *mm_result,
                                            const BIITensorInfo *vector_sum_col,
                                            const BIITensorInfo *vector_sum_row,
                                            int32_t a_offset,
                                            int32_t b_offset) {
                    BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mm_result, 1, BIDataType::S32, BIDataType::F32);

                    // We run if the offset is nonzero or a sum col has been provided, we need
                    // the second option in case the QuantizationInfo is dynamic
                    if (a_offset != 0 || vector_sum_col != nullptr) {
                        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_col, 1, BIDataType::S32);
                        BI_COMPUTE_RETURN_ERROR_ON(vector_sum_col->dimension(0) != mm_result->dimension(0));
                    }

                    // We run if the offset is nonzero or a sum row has been provided, we need
                    // the second option in case the QuantizationInfo is dynamic
                    if (b_offset != 0 || vector_sum_row != nullptr) {
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

                        BITensorShape output_shape = mm_result->tensor_shape();
                        if (output_shape.num_dimensions() > 1) {
                            const unsigned int output_batch_idx = reinterpret_as_3d ? 3 : 2;

                            BITensorShape vector_sum_row_shape = vector_sum_row->tensor_shape();
                            vector_sum_row_shape.collapse_from(1);
                            output_shape.collapse_from(output_batch_idx);

                            BI_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_row_shape[1] != output_shape[output_batch_idx],
                                                           "mm_result tensor must have the same number of batches of output tensor")
                            ;

                            if (vector_sum_col != nullptr) {
                                BITensorShape vector_sum_col_shape = vector_sum_col->tensor_shape();
                                vector_sum_col_shape.collapse_from(1);

                                BI_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_col_shape[1] != 1 &&
                                                               vector_sum_col_shape[1] != vector_sum_row_shape[1],
                                                               "vector_sum_col tensor must have the same number of batches of "
                                                               "vector_sum_row_shape or the number of batches must be set to 1")
                                ;
                            }
                        }
                    }

                    return BIStatus{};
                }

                void run_offset_contribution_float(const BIWindow &window,
                                                   BIITensor *mm_result,
                                                   const BIITensor *vector_sum_col,
                                                   const BIITensor *vector_sum_row,
                                                   int32_t a_offset,
                                                   int32_t b_offset,
                                                   int32_t k_offset,
                                                   float scale,
                                                   bool slide_vector_sum_col,
                                                   bool is_gemm3d) {
                    BIWindow collapsed_window = window.collapse_if_possible(window, BIWindow::DimZ);
                    collapsed_window.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                    const int height_input = is_gemm3d ? mm_result->info()->dimension(1) : 0;
                    const int depth_input = is_gemm3d ? mm_result->info()->dimension(2) : 1;

                    const int window_start_x = window.x().start();
                    const int window_end_x = window.x().end();
                    const int window_step_x = 16;

                    // if vector_sum_col is nullptr then stride_y is 0, else get stride_y
                    const size_t sum_col_stride_y = (vector_sum_col != nullptr)
                                                        ? (vector_sum_col->info()->strides_in_bytes().y())
                                                        : 0;
                    BIIterator mm_result_it(mm_result, collapsed_window);

                    if ((a_offset != 0) && (b_offset != 0) && (vector_sum_col != nullptr) &&
                        (vector_sum_row != nullptr)) // true, true
                    {
                        // Set window for vector_sum_col
                        BIWindow win_vector_sum_col(collapsed_window);
                        win_vector_sum_col.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_col.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));

                        // Set window for vector_sum_row
                        BIWindow win_vector_sum_row(collapsed_window);
                        win_vector_sum_row.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_row.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_row.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));

                        BIIterator vector_sum_col_it(vector_sum_col, win_vector_sum_col);
                        BIIterator vector_sum_row_it(vector_sum_row, win_vector_sum_row);

                        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

                        // Offset in case vector_sum_col is batched
                        const int vector_sum_col_batch_offset =
                                slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

                        execute_window_loop(
                            collapsed_window,
                            [&](const BICoordinates &id) {
                                const int batch_id = id.z() / depth_input;
                                const size_t batch_offset_col = batch_id * (sum_col_stride_y);
                                auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                    vector_sum_col_it.ptr() + batch_offset_col +
                                    batch_id * vector_sum_col_batch_offset);
                                auto mm_result_ptr = reinterpret_cast<float *>(mm_result_it.ptr());

                                // Compute the leftover term due to b_offset.
                                int32_t b_offset_term_s32 =
                                        *(reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() +
                                                                            batch_id * sum_row_stride_y) +
                                          id.y() + (id.z() % depth_input) * height_input);
                                b_offset_term_s32 *= b_offset;

                                const int32x4_t b_offset_term_s32_vec = vdupq_n_s32(b_offset_term_s32);

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                    // Compute the leftover term due to a_offset.
                                    int32x4x4_t a_offset_term_s32 = {
                                        {
                                            vld1q_s32(vector_sum_col_ptr + x + 0),
                                            vld1q_s32(vector_sum_col_ptr + x + 4),
                                            vld1q_s32(vector_sum_col_ptr + x + 8),
                                            vld1q_s32(vector_sum_col_ptr + x + 12)
                                        }
                                    };

                                    a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], a_offset);
                                    a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], a_offset);
                                    a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], a_offset);
                                    a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], a_offset);

                                    // Add a_offset_term_s32 and b_offset_term_s32
                                    int32x4x4_t offset_term_s32 = {
                                        {
                                            vdupq_n_s32(k_offset), vdupq_n_s32(k_offset), vdupq_n_s32(k_offset),
                                            vdupq_n_s32(k_offset)
                                        }
                                    };

                                    offset_term_s32.val[0] =
                                            vaddq_s32(offset_term_s32.val[0],
                                                      vaddq_s32(a_offset_term_s32.val[0], b_offset_term_s32_vec));
                                    offset_term_s32.val[1] =
                                            vaddq_s32(offset_term_s32.val[1],
                                                      vaddq_s32(a_offset_term_s32.val[1], b_offset_term_s32_vec));
                                    offset_term_s32.val[2] =
                                            vaddq_s32(offset_term_s32.val[2],
                                                      vaddq_s32(a_offset_term_s32.val[2], b_offset_term_s32_vec));
                                    offset_term_s32.val[3] =
                                            vaddq_s32(offset_term_s32.val[3],
                                                      vaddq_s32(a_offset_term_s32.val[3], b_offset_term_s32_vec));

                                    float32x4x4_t in_f32 = {
                                        {
                                            vld1q_f32(mm_result_ptr + x + 0), vld1q_f32(mm_result_ptr + x + 4),
                                            vld1q_f32(mm_result_ptr + x + 8), vld1q_f32(mm_result_ptr + x + 12)
                                        }
                                    };

                                    // Convert and scale the S32 offsets to match the already scaled GEMM results
                                    float32x4x4_t offset_terms_scaled = {
                                        {
                                            vmulq_n_f32(vcvtq_f32_s32(
                                                            offset_term_s32.val[0]),
                                                        scale),
                                            vmulq_n_f32(vcvtq_f32_s32(
                                                            offset_term_s32.val[1]),
                                                        scale),
                                            vmulq_n_f32(vcvtq_f32_s32(
                                                            offset_term_s32.val[2]),
                                                        scale),
                                            vmulq_n_f32(vcvtq_f32_s32(
                                                            offset_term_s32.val[3]),
                                                        scale),
                                        }
                                    };

                                    // Add the offset terms to the GEMM result
                                    in_f32.val[0] = vaddq_f32(in_f32.val[0], offset_terms_scaled.val[0]);
                                    in_f32.val[1] = vaddq_f32(in_f32.val[1], offset_terms_scaled.val[1]);
                                    in_f32.val[2] = vaddq_f32(in_f32.val[2], offset_terms_scaled.val[2]);
                                    in_f32.val[3] = vaddq_f32(in_f32.val[3], offset_terms_scaled.val[3]);

                                    // Store the result with the offset contribution
                                    vst1q_f32(mm_result_ptr + x + 0, in_f32.val[0]);
                                    vst1q_f32(mm_result_ptr + x + 4, in_f32.val[1]);
                                    vst1q_f32(mm_result_ptr + x + 8, in_f32.val[2]);
                                    vst1q_f32(mm_result_ptr + x + 12, in_f32.val[3]);
                                }

                                // Left-overs loop
                                for (; x < window_end_x; ++x) {
                                    // Compute the leftover term due to a_offset.
                                    int32_t a_offset_term_s32 = *(vector_sum_col_ptr + x);

                                    a_offset_term_s32 *= a_offset;

                                    // Add the offset terms to GEMM's result
                                    // Store the result with the offset contribution
                                    mm_result_ptr[x] += (k_offset + a_offset_term_s32 + b_offset_term_s32) * scale;
                                }
                            },
                            vector_sum_col_it, vector_sum_row_it, mm_result_it);
                    } else if ((a_offset == 0) && (b_offset != 0) && (vector_sum_row != nullptr)) // false, true
                    {
                        BI_COMPUTE_ERROR_ON_NULLPTR(vector_sum_row);

                        // Set window for vector_sum_row
                        BIWindow win_vector_sum_row(collapsed_window);
                        win_vector_sum_row.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_row.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_row.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));

                        BIIterator vector_sum_row_it(vector_sum_row, win_vector_sum_row);

                        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

                        execute_window_loop(
                            collapsed_window,
                            [&](const BICoordinates &id) {
                                const int batch_id = id.z() / depth_input;
                                auto mm_result_ptr = reinterpret_cast<float *>(mm_result_it.ptr());

                                // Compute the leftover term due to b_offset.
                                int32_t row_sum =
                                        *(reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() +
                                                                            batch_id * sum_row_stride_y) +
                                          id.y() + (id.z() % depth_input) * height_input);
                                float scaled_b_offset_term_f32 = row_sum * b_offset * scale;

                                const float32x4_t b_offset_term_f32_vec = vdupq_n_f32(scaled_b_offset_term_f32);

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                    float32x4x4_t in_f32 = {
                                        {
                                            vld1q_f32(mm_result_ptr + x + 0), vld1q_f32(mm_result_ptr + x + 4),
                                            vld1q_f32(mm_result_ptr + x + 8), vld1q_f32(mm_result_ptr + x + 12)
                                        }
                                    };

                                    // Add the offset terms to GEMM's result
                                    in_f32.val[0] = vaddq_f32(in_f32.val[0], b_offset_term_f32_vec);
                                    in_f32.val[1] = vaddq_f32(in_f32.val[1], b_offset_term_f32_vec);
                                    in_f32.val[2] = vaddq_f32(in_f32.val[2], b_offset_term_f32_vec);
                                    in_f32.val[3] = vaddq_f32(in_f32.val[3], b_offset_term_f32_vec);

                                    // Store the result with the offset contribution
                                    vst1q_f32(mm_result_ptr + x + 0, in_f32.val[0]);
                                    vst1q_f32(mm_result_ptr + x + 4, in_f32.val[1]);
                                    vst1q_f32(mm_result_ptr + x + 8, in_f32.val[2]);
                                    vst1q_f32(mm_result_ptr + x + 12, in_f32.val[3]);
                                }

                                // Left-overs loop
                                for (; x < window_end_x; ++x) {
                                    // Add the offset terms to GEMM's result
                                    // Store the result with the offset contribution
                                    mm_result_ptr[x] += scaled_b_offset_term_f32;
                                }
                            },
                            vector_sum_row_it, mm_result_it);
                    } else if ((a_offset != 0) && (b_offset == 0) && (vector_sum_col != nullptr)) // true, false
                    {
                        // Set window for vector_sum_col
                        BIWindow win_vector_sum_col(collapsed_window);
                        win_vector_sum_col.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_col.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));

                        BIIterator vector_sum_col_it(vector_sum_col, win_vector_sum_col);

                        // Offset in case vector_sum_col is batched
                        const int vector_sum_col_batch_offset =
                                slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

                        execute_window_loop(
                            collapsed_window,
                            [&](const BICoordinates &id) {
                                const int batch_id = id.z() / depth_input;
                                const size_t batch_offset_col =
                                        batch_id *
                                        (sum_col_stride_y);
                                // Value to offset vector_sum_col_ptr to allow for iteration of y values in tensor
                                auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                    vector_sum_col_it.ptr() + batch_offset_col +
                                    batch_id * vector_sum_col_batch_offset);
                                auto mm_result_ptr = reinterpret_cast<float *>(mm_result_it.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                    // Compute the leftover term due to a_offset.
                                    int32x4x4_t a_offset_term_s32 = {
                                        {
                                            vld1q_s32(vector_sum_col_ptr + x + 0),
                                            vld1q_s32(vector_sum_col_ptr + x + 4),
                                            vld1q_s32(vector_sum_col_ptr + x + 8),
                                            vld1q_s32(vector_sum_col_ptr + x + 12)
                                        }
                                    };

                                    a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], a_offset);
                                    a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], a_offset);
                                    a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], a_offset);
                                    a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], a_offset);

                                    float32x4x4_t a_offset_term_scaled = {
                                        {
                                            vmulq_n_f32(vcvtq_f32_s32(
                                                            a_offset_term_s32.val[0]),
                                                        scale),
                                            vmulq_n_f32(vcvtq_f32_s32(
                                                            a_offset_term_s32.val[1]),
                                                        scale),
                                            vmulq_n_f32(vcvtq_f32_s32(
                                                            a_offset_term_s32.val[2]),
                                                        scale),
                                            vmulq_n_f32(vcvtq_f32_s32(
                                                            a_offset_term_s32.val[3]),
                                                        scale),
                                        }
                                    };

                                    float32x4x4_t in_f32 = {
                                        {
                                            vld1q_f32(mm_result_ptr + x + 0), vld1q_f32(mm_result_ptr + x + 4),
                                            vld1q_f32(mm_result_ptr + x + 8), vld1q_f32(mm_result_ptr + x + 12)
                                        }
                                    };

                                    // Add the offset terms to GEMM's result
                                    in_f32.val[0] = vaddq_f32(in_f32.val[0], a_offset_term_scaled.val[0]);
                                    in_f32.val[1] = vaddq_f32(in_f32.val[1], a_offset_term_scaled.val[1]);
                                    in_f32.val[2] = vaddq_f32(in_f32.val[2], a_offset_term_scaled.val[2]);
                                    in_f32.val[3] = vaddq_f32(in_f32.val[3], a_offset_term_scaled.val[3]);

                                    // Store the result with the offset contribution
                                    vst1q_f32(mm_result_ptr + x + 0, in_f32.val[0]);
                                    vst1q_f32(mm_result_ptr + x + 4, in_f32.val[1]);
                                    vst1q_f32(mm_result_ptr + x + 8, in_f32.val[2]);
                                    vst1q_f32(mm_result_ptr + x + 12, in_f32.val[3]);
                                }

                                // Left-overs loop
                                for (; x < window_end_x; ++x) {
                                    // Compute the leftover term due to a_offset.
                                    const int32_t a_offset_term_s32 = *(vector_sum_col_ptr + x);

                                    // Add the offset terms to GEMM's result
                                    // Store the result with the offset contribution
                                    mm_result_ptr[x] += a_offset_term_s32 * a_offset * scale;
                                }
                            },
                            vector_sum_col_it, mm_result_it);
                    } else // false, false
                    {
                        // No offset contribution from matrix A and matrix B
                        return;
                    }
                }

                void run_offset_contribution(const BIWindow &window,
                                             BIITensor *mm_result,
                                             const BIITensor *vector_sum_col,
                                             const BIITensor *vector_sum_row,
                                             int32_t a_offset,
                                             int32_t b_offset,
                                             int32_t k_offset,
                                             bool slide_vector_sum_col,
                                             bool is_gemm3d) {
                    BIWindow collapsed_window = window.collapse_if_possible(window, BIWindow::DimZ);
                    collapsed_window.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));

                    const int height_input = is_gemm3d ? mm_result->info()->dimension(1) : 0;
                    const int depth_input = is_gemm3d ? mm_result->info()->dimension(2) : 1;

                    const int window_start_x = window.x().start();
                    const int window_end_x = window.x().end();
                    const int window_step_x = 16;

                    // if vector_sum_col is nullptr then stride_y is 0, else get stride_y
                    const size_t sum_col_stride_y = (vector_sum_col != nullptr)
                                                        ? (vector_sum_col->info()->strides_in_bytes().y())
                                                        : 0;
                    BIIterator mm_result_it(mm_result, collapsed_window);

                    if ((a_offset != 0) && (b_offset != 0) && (vector_sum_col != nullptr) &&
                        (vector_sum_row != nullptr)) // true, true
                    {
                        // Set window for vector_sum_col
                        BIWindow win_vector_sum_col(collapsed_window);
                        win_vector_sum_col.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_col.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));

                        // Set window for vector_sum_row
                        BIWindow win_vector_sum_row(collapsed_window);
                        win_vector_sum_row.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_row.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_row.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));

                        BIIterator vector_sum_col_it(vector_sum_col, win_vector_sum_col);
                        BIIterator vector_sum_row_it(vector_sum_row, win_vector_sum_row);

                        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

                        // Offset in case vector_sum_col is batched
                        const int vector_sum_col_batch_offset =
                                slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

                        execute_window_loop(
                            collapsed_window,
                            [&](const BICoordinates &id) {
                                const int batch_id = id.z() / depth_input;
                                const size_t batch_offset_col = batch_id * (sum_col_stride_y);
                                auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                    vector_sum_col_it.ptr() + batch_offset_col +
                                    batch_id * vector_sum_col_batch_offset);
                                auto mm_result_ptr = reinterpret_cast<int32_t *>(mm_result_it.ptr());

                                // Compute the leftover term due to b_offset.
                                int32_t b_offset_term_s32 =
                                        *(reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() +
                                                                            batch_id * sum_row_stride_y) +
                                          id.y() + (id.z() % depth_input) * height_input);
                                b_offset_term_s32 *= b_offset;

                                const int32x4_t b_offset_term_s32_vec = vdupq_n_s32(b_offset_term_s32);

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                    // Compute the leftover term due to a_offset.
                                    int32x4x4_t a_offset_term_s32 = {
                                        {
                                            vld1q_s32(vector_sum_col_ptr + x + 0),
                                            vld1q_s32(vector_sum_col_ptr + x + 4),
                                            vld1q_s32(vector_sum_col_ptr + x + 8),
                                            vld1q_s32(vector_sum_col_ptr + x + 12)
                                        }
                                    };

                                    a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], a_offset);
                                    a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], a_offset);
                                    a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], a_offset);
                                    a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], a_offset);

                                    // Add a_offset_term_s32 and b_offset_term_s32
                                    int32x4x4_t offset_term_s32 = {
                                        {
                                            vdupq_n_s32(k_offset), vdupq_n_s32(k_offset), vdupq_n_s32(k_offset),
                                            vdupq_n_s32(k_offset)
                                        }
                                    };

                                    offset_term_s32.val[0] =
                                            vaddq_s32(offset_term_s32.val[0],
                                                      vaddq_s32(a_offset_term_s32.val[0], b_offset_term_s32_vec));
                                    offset_term_s32.val[1] =
                                            vaddq_s32(offset_term_s32.val[1],
                                                      vaddq_s32(a_offset_term_s32.val[1], b_offset_term_s32_vec));
                                    offset_term_s32.val[2] =
                                            vaddq_s32(offset_term_s32.val[2],
                                                      vaddq_s32(a_offset_term_s32.val[2], b_offset_term_s32_vec));
                                    offset_term_s32.val[3] =
                                            vaddq_s32(offset_term_s32.val[3],
                                                      vaddq_s32(a_offset_term_s32.val[3], b_offset_term_s32_vec));

                                    int32x4x4_t in_s32 = {
                                        {
                                            vld1q_s32(mm_result_ptr + x + 0), vld1q_s32(mm_result_ptr + x + 4),
                                            vld1q_s32(mm_result_ptr + x + 8), vld1q_s32(mm_result_ptr + x + 12)
                                        }
                                    };

                                    // Add the offset terms to GEMM's result
                                    in_s32.val[0] = vaddq_s32(in_s32.val[0], offset_term_s32.val[0]);
                                    in_s32.val[1] = vaddq_s32(in_s32.val[1], offset_term_s32.val[1]);
                                    in_s32.val[2] = vaddq_s32(in_s32.val[2], offset_term_s32.val[2]);
                                    in_s32.val[3] = vaddq_s32(in_s32.val[3], offset_term_s32.val[3]);

                                    // Store the result with the offset contribution
                                    vst1q_s32(mm_result_ptr + x + 0, in_s32.val[0]);
                                    vst1q_s32(mm_result_ptr + x + 4, in_s32.val[1]);
                                    vst1q_s32(mm_result_ptr + x + 8, in_s32.val[2]);
                                    vst1q_s32(mm_result_ptr + x + 12, in_s32.val[3]);
                                }

                                // Left-overs loop
                                for (; x < window_end_x; ++x) {
                                    // Compute the leftover term due to a_offset.
                                    int32_t a_offset_term_s32 = *(vector_sum_col_ptr + x);

                                    a_offset_term_s32 *= a_offset;

                                    // Add the offset terms to GEMM's result
                                    // Store the result with the offset contribution
                                    mm_result_ptr[x] += k_offset + a_offset_term_s32 + b_offset_term_s32;
                                }
                            },
                            vector_sum_col_it, vector_sum_row_it, mm_result_it);
                    } else if ((a_offset == 0) && (b_offset != 0) && (vector_sum_row != nullptr)) // false, true
                    {
                        BI_COMPUTE_ERROR_ON_NULLPTR(vector_sum_row);

                        // Set window for vector_sum_row
                        BIWindow win_vector_sum_row(collapsed_window);
                        win_vector_sum_row.set(BIWindow::DimX, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_row.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_row.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));

                        BIIterator vector_sum_row_it(vector_sum_row, win_vector_sum_row);

                        const size_t sum_row_stride_y = vector_sum_row->info()->strides_in_bytes().y();

                        execute_window_loop(
                            collapsed_window,
                            [&](const BICoordinates &id) {
                                const int batch_id = id.z() / depth_input;
                                auto mm_result_ptr = reinterpret_cast<int32_t *>(mm_result_it.ptr());

                                // Compute the leftover term due to b_offset.
                                int32_t b_offset_term_s32 =
                                        *(reinterpret_cast<const int32_t *>(vector_sum_row_it.ptr() +
                                                                            batch_id * sum_row_stride_y) +
                                          id.y() + (id.z() % depth_input) * height_input);
                                b_offset_term_s32 *= b_offset;

                                const int32x4_t b_offset_term_s32_vec = vdupq_n_s32(b_offset_term_s32);

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                    int32x4x4_t in_s32 = {
                                        {
                                            vld1q_s32(mm_result_ptr + x + 0), vld1q_s32(mm_result_ptr + x + 4),
                                            vld1q_s32(mm_result_ptr + x + 8), vld1q_s32(mm_result_ptr + x + 12)
                                        }
                                    };

                                    // Add the offset terms to GEMM's result
                                    in_s32.val[0] = vaddq_s32(in_s32.val[0], b_offset_term_s32_vec);
                                    in_s32.val[1] = vaddq_s32(in_s32.val[1], b_offset_term_s32_vec);
                                    in_s32.val[2] = vaddq_s32(in_s32.val[2], b_offset_term_s32_vec);
                                    in_s32.val[3] = vaddq_s32(in_s32.val[3], b_offset_term_s32_vec);

                                    // Store the result with the offset contribution
                                    vst1q_s32(mm_result_ptr + x + 0, in_s32.val[0]);
                                    vst1q_s32(mm_result_ptr + x + 4, in_s32.val[1]);
                                    vst1q_s32(mm_result_ptr + x + 8, in_s32.val[2]);
                                    vst1q_s32(mm_result_ptr + x + 12, in_s32.val[3]);
                                }

                                // Left-overs loop
                                for (; x < window_end_x; ++x) {
                                    // Add the offset terms to GEMM's result
                                    // Store the result with the offset contribution
                                    mm_result_ptr[x] += b_offset_term_s32;
                                }
                            },
                            vector_sum_row_it, mm_result_it);
                    } else if ((a_offset != 0) && (b_offset == 0) && (vector_sum_col != nullptr)) // true, false
                    {
                        // Set window for vector_sum_col
                        BIWindow win_vector_sum_col(collapsed_window);
                        win_vector_sum_col.set(BIWindow::DimY, BIWindow::BIDimension(0, 0, 0));
                        win_vector_sum_col.set(BIWindow::DimZ, BIWindow::BIDimension(0, 0, 0));

                        BIIterator vector_sum_col_it(vector_sum_col, win_vector_sum_col);

                        // Offset in case vector_sum_col is batched
                        const int vector_sum_col_batch_offset =
                                slide_vector_sum_col ? vector_sum_col->info()->strides_in_bytes().z() : 0;

                        execute_window_loop(
                            collapsed_window,
                            [&](const BICoordinates &id) {
                                const int batch_id = id.z() / depth_input;
                                const size_t batch_offset_col = batch_id * (sum_col_stride_y);
                                // Value to offset vector_sum_col_ptr to allow for iteration of y values in tensor
                                // TODO: 目前替换掉官方的这个代码，因为这个模块计算的offset不需要进行vector位移，只有[0 ~ sum_col_stride_y]的长度
                                // auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                //     vector_sum_col_it.ptr() + batch_offset_col +
                                //     batch_id * vector_sum_col_batch_offset);
                                auto vector_sum_col_ptr = reinterpret_cast<const int32_t *>(
                                    vector_sum_col_it.ptr() + batch_id * vector_sum_col_batch_offset);
                                auto mm_result_ptr = reinterpret_cast<int32_t *>(mm_result_it.ptr());

                                int x = window_start_x;
                                for (; x <= (window_end_x - window_step_x); x += window_step_x) {
                                    // Compute the leftover term due to a_offset.
                                    int32x4x4_t a_offset_term_s32 = {
                                        {
                                            vld1q_s32(vector_sum_col_ptr + x + 0),
                                            vld1q_s32(vector_sum_col_ptr + x + 4),
                                            vld1q_s32(vector_sum_col_ptr + x + 8),
                                            vld1q_s32(vector_sum_col_ptr + x + 12)
                                        }
                                    };

                                    a_offset_term_s32.val[0] = vmulq_n_s32(a_offset_term_s32.val[0], a_offset);
                                    a_offset_term_s32.val[1] = vmulq_n_s32(a_offset_term_s32.val[1], a_offset);
                                    a_offset_term_s32.val[2] = vmulq_n_s32(a_offset_term_s32.val[2], a_offset);
                                    a_offset_term_s32.val[3] = vmulq_n_s32(a_offset_term_s32.val[3], a_offset);

                                    int32x4x4_t in_s32 = {
                                        {
                                            vld1q_s32(mm_result_ptr + x + 0), vld1q_s32(mm_result_ptr + x + 4),
                                            vld1q_s32(mm_result_ptr + x + 8), vld1q_s32(mm_result_ptr + x + 12)
                                        }
                                    };

                                    // Add the offset terms to GEMM's result
                                    in_s32.val[0] = vaddq_s32(in_s32.val[0], a_offset_term_s32.val[0]);
                                    in_s32.val[1] = vaddq_s32(in_s32.val[1], a_offset_term_s32.val[1]);
                                    in_s32.val[2] = vaddq_s32(in_s32.val[2], a_offset_term_s32.val[2]);
                                    in_s32.val[3] = vaddq_s32(in_s32.val[3], a_offset_term_s32.val[3]);

                                    // Store the result with the offset contribution
                                    vst1q_s32(mm_result_ptr + x + 0, in_s32.val[0]);
                                    vst1q_s32(mm_result_ptr + x + 4, in_s32.val[1]);
                                    vst1q_s32(mm_result_ptr + x + 8, in_s32.val[2]);
                                    vst1q_s32(mm_result_ptr + x + 12, in_s32.val[3]);
                                }

                                // Left-overs loop
                                for (; x < window_end_x; ++x) {
                                    // Compute the leftover term due to a_offset.
                                    const int32_t a_offset_term_s32 = *(vector_sum_col_ptr + x);

                                    // Add the offset terms to GEMM's result
                                    // Store the result with the offset contribution
                                    mm_result_ptr[x] += a_offset_term_s32 * a_offset;
                                }
                            },
                            vector_sum_col_it, mm_result_it);
                    } else // false, false
                    {
                        // No offset contribution from matrix A and matrix B
                        return;
                    }
                }
            }

            void BICpuGemmLowpOffsetContributionKernel::configure(BIITensorInfo *mm_result,
                                                                  BIITensorInfo *vector_sum_col,
                                                                  BIITensorInfo *vector_sum_row,
                                                                  int32_t k,
                                                                  int32_t a_offset,
                                                                  int32_t b_offset,
                                                                  float scale) {
                // Perform validate step
                BI_COMPUTE_UNUSED(vector_sum_row);
                BI_COMPUTE_ERROR_ON_NULLPTR(mm_result);
                BI_COMPUTE_ERROR_THROW_ON(
                    validate_arguments(mm_result, vector_sum_col, vector_sum_row, a_offset, b_offset));

                _a_offset = a_offset;
                _b_offset = b_offset;
                _k = k;

                _scale = scale;

                if (vector_sum_col != nullptr) {
                    // Check if vector_sum_col_shape should be slidden or not
                    // Don't slide vector_sum_col_shape along the y dimension if vector_sum_col_shape has just 1 dimension and vector_sum_row_shape more than 1
                    // This scenario can happen when the matrix multiplication is used to perform a convolution operation
                    _slide_vector_sum_col = vector_sum_col->tensor_shape().num_dimensions() > 1;
                }

                // Configure kernel window
                BIWindow win = calculate_max_window(*mm_result, BISteps());
                BIICpuKernel::configure(win);
            }

            void BICpuGemmLowpOffsetContributionKernel::dynamic_configure(BIITensorInfo *mm_result) {
                auto win = BIICpuKernel::window();
                dynamic_calculate_max_window(mm_result->tensor_shape(), BISteps(), win);
                BIICpuKernel::dynamic_configure(win);
            }


            void BICpuGemmLowpOffsetContributionKernel::set_a_offset(int32_t a_offset) {
                _a_offset = a_offset;
            }

            void BICpuGemmLowpOffsetContributionKernel::set_b_offset(int32_t b_offset) {
                _b_offset = b_offset;
            }

            void BICpuGemmLowpOffsetContributionKernel::set_scale(float scale) {
                _scale = scale;
            }

            BIStatus BICpuGemmLowpOffsetContributionKernel::validate(const BIITensorInfo *mm_result,
                                                                     const BIITensorInfo *vector_sum_col,
                                                                     const BIITensorInfo *vector_sum_row,
                                                                     int32_t a_offset,
                                                                     int32_t b_offset) {
                BI_COMPUTE_RETURN_ON_ERROR(
                    validate_arguments(mm_result, vector_sum_col, vector_sum_row, a_offset, b_offset));
                return BIStatus{};
            }

            void BICpuGemmLowpOffsetContributionKernel::run_op(BIITensorPack &tensors, const BIWindow &window,
                                                               const ThreadInfo &info) {
                BI_COMPUTE_UNUSED(info);
                BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
                BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIICpuKernel::window(), window);

                auto vector_sum_col = tensors.get_const_tensor(BITensorType::ACL_SRC_0);
                auto vector_sum_row = tensors.get_const_tensor(BITensorType::ACL_SRC_1);
                auto mm_result = tensors.get_tensor(BITensorType::ACL_DST);

                // Check if input is a 3D reinterpretation
                const bool reinterpret_as_3d = vector_sum_row != nullptr && mm_result->info()->num_dimensions() > 1 &&
                                               mm_result->info()->tensor_shape().y() !=
                                               vector_sum_row->info()->tensor_shape().x();

                // check to see what is the output type of result
                auto k_offset = _a_offset * _b_offset * _k;
                if (mm_result->info()->data_type() == BIDataType::F32) {
                    run_offset_contribution_float(window, mm_result, vector_sum_col, vector_sum_row, _a_offset,
                                                  _b_offset, k_offset,
                                                  _scale, _slide_vector_sum_col, reinterpret_as_3d);
                } else {
                    run_offset_contribution(window, mm_result, vector_sum_col, vector_sum_row, _a_offset, _b_offset,
                                            k_offset,
                                            _slide_vector_sum_col, reinterpret_as_3d);
                }
            }

            const char *BICpuGemmLowpOffsetContributionKernel::name() const {
                return "BICpuGemmLowpOffsetContributionKernel";
            }
        }
    }
}
