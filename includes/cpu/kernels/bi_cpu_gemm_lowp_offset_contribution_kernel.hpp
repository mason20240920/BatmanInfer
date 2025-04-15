//
// Created by Mason on 2025/1/20.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

#include <cstdint>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /** Kernel used to add the offset contribution after @ref CpuGemmLowpMatrixMultiplyKernel. The computation is performed in-place
             *
             * This kernel takes a final int32 accumulator value (the output of @ref CpuGemmLowpMatrixMultiplyKernel),
             * and adds to it the offset contribution of matrix A and matrix B in-place.
             *
             * The final result is:
             *
             * mm_result[i][k] = mm_result[i][k] +
             *                   (vector_sum_col[k] * a_offset) +
             *                   (vector_sum_row[i] * b_offset) +
             *                   (a_offset * b_offset * k)
             *
             */
            class BICpuGemmLowpOffsetContributionKernel : public BIICpuKernel<BICpuGemmLowpOffsetContributionKernel> {
            public:
                /** Default constructor */
                BICpuGemmLowpOffsetContributionKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmLowpOffsetContributionKernel);

                /** Initialise the kernel's input and output.
                *
                * @param[in, out] mm_result      Input tensor containing the result of @ref CpuGemmLowpMatrixMultiplyKernel. Data type supported: S32
                * @param[in]      vector_sum_col Input row-vector of sums of all the entries in each column of matrix B.
                *                                Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
                * @param[in]      vector_sum_row Input row-vector of sums of all the entries in each row of matrix A.
                *                                Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: same as @p mm_result
                * @param[in]      k              Number of matrix A columns or Matrix B rows
                * @param[in]      a_offset       Offset to be added to each element of the matrix A.
                * @param[in]      b_offset       Offset to be added to each element of the matrix B.
                * @param[in]      scale          (Optional) multiplies the contribution to make it the same scale as the dst in the case where mm_result is float
                *                                (and so has already been scaled). Default is 1.0
                */
                void configure(BIITensorInfo *mm_result,
                               BIITensorInfo *vector_sum_col,
                               BIITensorInfo *vector_sum_row,
                               int32_t k,
                               int32_t a_offset,
                               int32_t b_offset,
                               float scale = 1.0f);

                void dynamic_configure(BIITensorInfo *mm_result);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to CpuGemmLowpOffsetContributionKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus validate(const BIITensorInfo *mm_result,
                                         const BIITensorInfo *vector_sum_col,
                                         const BIITensorInfo *vector_sum_row,
                                         int32_t a_offset,
                                         int32_t b_offset);

                /** Set the a offset
                 * Warning: if a_offset is non-zero then vector_sum_col must be set in run_op.
                 *          Run configure or validate again if you aren't sure
                 *
                 * @param[in] a_offset Offset to be added to each element of the matrix A.
                 */
                void set_a_offset(int32_t a_offset);

                /** Set the b offset
                 * Warning: if b_offset is non-zero then vector_sum_row must be set in run_op.
                 *          Run configure or validate again if you aren't sure
                 *
                 * @param[in] b_offset Offset to be added to each element of the matrix B.
                 */
                void set_b_offset(int32_t b_offset);

                /** Set the dequantize scale
                 *
                 * @param[in] scale Multiplies the contribution to make it the same scale as the dst in the case where
                 *                  mm_result is float (and so has already been scaled).
                 */
                void set_scale(float scale);

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

            private:
                int32_t _a_offset{0};
                int32_t _b_offset{0};
                int32_t _k{0}; // Number of columns of A or rows of B, used in last offset term
                float _scale{1.0};
                bool _slide_vector_sum_col{true};
            };
        }
    }
}
