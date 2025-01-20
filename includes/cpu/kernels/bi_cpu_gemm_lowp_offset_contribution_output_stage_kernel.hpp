//
// Created by Mason on 2025/1/20.
//

#pragma once

#include <data/core/kernel_descriptors.hpp>

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /**
             * 内核用于在执行完 CpuGemmLowpMatrixMultiplyKernel 后，添加偏移贡献并完成输出阶段的量化操作。
             * 计算是就地（in-place）进行的。
             *
             * 主要功能：
             * 1. 偏移修正：对矩阵乘法的结果 mm_result[i][k] 添加矩阵 A 和矩阵 B 的偏移贡献，得到修正后的结果 mm_result'[i][k]。
             * 2. 量化输出：将修正后的结果量化为 uint8 或 int8 类型，适配神经网络的推理需求。
             *
             * 偏移修正公式：
             * mm_result'[i][k] = mm_result[i][k] +
             *                    (vector_sum_col[k] * a_offset) +    // 矩阵 B 每列的和乘以矩阵 A 的零点
             *                    (vector_sum_row[i] * b_offset) +    // 矩阵 A 每行的和乘以矩阵 B 的零点
             *                    (a_offset * b_offset * k)          // A 和 B 的零点乘积再乘以列数
             *
             * 量化输出公式：
             * 1. 对于整数乘法量化（QuantizeDownInt32ToUint8Scale 或 QuantizeDownInt32ToInt8Scale）：
             *    final_result[i][k] = ((mm_result'[i][k] + result_offset) * result_mult_int) >> result_shift
             *
             * 2. 对于定点乘法量化（QuantizeDownInt32ToUint8ScaleByFixedPoint 或 QuantizeDownInt32ToInt8ScaleByFixedPoint）：
             *    final_result[i][k] = (FixedPointMul(mm_result'[i][k], result_fixedpoint_multiplier) >> result_shift)
             *                         + result_offset_after_shift
             *
             * 定点乘法（FixedPointMul）：
             * FixedPointMul(x, y) = (x * y) / 2^31 的最接近整数值，计算时无溢出或中间舍入。
             *
             * 示例：
             * 假设：
             * - mm_result[i][k] = 1000
             * - vector_sum_col[k] = 20, vector_sum_row[i] = 15
             * - a_offset = 3, b_offset = 5, k = 10
             * 则：
             * mm_result'[i][k] = 1000 + (20 * 3) + (15 * 5) + (3 * 5 * 10)
             *                  = 1000 + 60 + 75 + 150 = 1285
             *
             * 如果使用整数乘法量化：
             * - result_offset = 128, result_mult_int = 2, result_shift = 1
             * 则：
             * final_result[i][k] = ((1285 + 128) * 2) >> 1
             *                    = (1413 * 2) >> 1 = 2826 >> 1 = 1413
             *
             * 最终，结果被量化为 uint8 或 int8 类型。
             */

            class BICpuGemmLowpOffsetContributionOutputStageKernel
                    : public BIICpuKernel<BICpuGemmLowpOffsetContributionOutputStageKernel> {
            public:
                /** Default constructor */
                BICpuGemmLowpOffsetContributionOutputStageKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmLowpOffsetContributionOutputStageKernel);

                /** Initialise the kernel inputs and output.
                 *
                 * @param[in]  mm_result      Input tensor info containing the result of @ref CpuGemmLowpMatrixMultiplyKernel. Data type supported: S32
                 * @param[in]  vector_sum_col Input row-vector tensor info of sums of all the entries in each column of matrix B.
                 *                            Can be a 1D or 2D tensor, in case of 2D, y dim is the batch dimension
                 *                            Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
                 * @param[in]  vector_sum_row Input row-vector tensor info of sums of all the entries in each row of matrix A.
                 *                            Can be a 1D or 2D tensor, in case of 2D, y dim is the batch dimension
                 * @param[in]  bias           Biases tensor info. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
                 *                            Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p mm_result.
                 * @param[out] dst            Output tensor info containing the final quantized result. Data type supported: QASYMM8/QASYMM8_SIGNED
                 * @param[in]  k              Number of matrix A columns or Matrix B rows
                 * @param[in]  a_offset       Offset to be added to each element of the matrix A.
                 * @param[in]  b_offset       Offset to be added to each element of the matrix B.
                 * @param[in]  output_stage   GEMMLowp output stage info, providing the type of quantization and the necessary parameters.
                 */
                void configure(const BIITensorInfo *mm_result,
                               const BIITensorInfo *vector_sum_col,
                               const BIITensorInfo *vector_sum_row,
                               const BIITensorInfo *bias,
                               BIITensorInfo *dst,
                               int32_t k,
                               int32_t a_offset,
                               int32_t b_offset,
                               BIGEMMLowpOutputStageInfo output_stage);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to CpuGemmLowpOffsetContributionOutputStageKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus validate(const BIITensorInfo *mm_result,
                                         const BIITensorInfo *vector_sum_col,
                                         const BIITensorInfo *vector_sum_row,
                                         const BIITensorInfo *bias,
                                         const BIITensorInfo *dst,
                                         int32_t a_offset,
                                         int32_t b_offset,
                                         BIGEMMLowpOutputStageInfo output_stage);

                /** Set the a offset
                 * Warning: if a_offset is non-zero then vector_sum_col must be set in run_op.
                 *          Run configure or validate again if you aren't sure
                 *
                 * @param[in] a_offset Offset to be added to each element of the matrix A.
                 */
                void set_a_offset(int32_t a_offset);

                /** Set the b offset
                 * Warning: if b_offset is non-zero then vector_sum_col must be set in run_op.
                 *          Run configure or validate again if you aren't sure
                 *
                 * @param[in] b_offset Offset to be added to each element of the matrix B.
                 */
                void set_b_offset(int32_t b_offset);

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

            private:
                /** Function to use for the particular tensors passed to configure() */
                int32_t _a_offset{0};
                int32_t _b_offset{0};
                int32_t _k{0}; // Number of columns of A or rows of B, used in last offset term
                bool _is_vector_sum_col_batched{true};
                BIGEMMLowpOutputStageInfo _output_stage{BIGEMMLowpOutputStageInfo()};
            };
        }
    }
}