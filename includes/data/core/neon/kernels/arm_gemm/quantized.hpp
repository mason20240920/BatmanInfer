//
// Created by Mason on 2025/1/14.
//

#pragma once

#include "utils.hpp" // IndirectInputArg

namespace BatmanGemm {

    template<typename Tin, typename Tout>
    void requantize_block_32(const Requantize32 &qp, unsigned int width, unsigned int height,
                             const Tin *input, unsigned int in_stride, Tout *output, unsigned int out_stride,
                             const int32_t *row_bias, const int32_t *col_bias, unsigned int start_col);

    template<typename T>
    void compute_row_sums(const Requantize32 &qp, unsigned int width, unsigned int height,
                          const T *input, unsigned int in_stride, int32_t *row_bias);

    template<typename T>
    void compute_col_sums(const Requantize32 &qp, unsigned int width, unsigned int height,
                          const T *input, unsigned int in_stride, int32_t *col_bias, unsigned int depth,
                          unsigned int multi, unsigned int first_col);

    template<typename T>
    void row_sums_indirect(size_t num_strings, const unsigned int *string_lengths, IndirectInputArg<T> A_arg,
                           size_t M, int32_t *output_ptr, const Requantize32 *qp);

    void dequantize_block_32(const DequantizeFloat &qp, unsigned int width, unsigned int height,
                             const int32_t *input, unsigned int in_stride, float *output, unsigned int out_stride,
                             const float *row_bias, bool not_first_pass, const Activation &act);

} // namespace arm_gemm