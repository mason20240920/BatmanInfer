//
// Created by Mason on 2025/1/20.
//

#pragma once

#include "data/core/bi_tensor_info.hpp"
#include "function_info/bi_GEMMInfo.h"

#include "data/core/common/bi_core_common_macros.hpp"
#include "cpu/bi_i_cpu_operator.hpp"

#include <memory>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            class BICpuGemmInterleave4x4Kernel;

            class BICpuGemmLowpMatrixMultiplyKernel;

            class BICpuGemmLowpOffsetContributionKernel;

            class BICpuGemmLowpOffsetContributionOutputStageKernel;

            class BICpuGemmLowpMatrixAReductionKernel;

            class BICpuGemmLowpMatrixBReductionKernel;

            class BICpuGemmTranspose1xWKernel;

            class BICpuConvertQuantizedSignednessKernel;
        } // namespace kernels
        class BICpuGemmAssemblyDispatch;

        class BICpuActivation;

        /** Basic function to execute BIGEMMLowpMatrixMultiplyCore. This function calls the following kernels if the DOT product instruction is not available:
         *
         *  -# @ref kernels::BICpuGemmInterleave4x4Kernel
         *  -# @ref kernels::BICpuGemmTranspose1xWKernel
         *  -# @ref kernels::BICpuGemmLowpMatrixMultiplyKernel
         *  -# @ref kernels::BICpuGemmLowpOffsetContributionKernel
         *  -# @ref BICpuActivation
         *
         * otherwise if the DOT product instruction is available:
         *
         *  -# @ref kernels::BICpuGemmLowpOffsetContributionKernel
         *
        */
        class BICpuGemmLowpMatrixMultiplyCore : public BIICpuOperator {
        public:
            BICpuGemmLowpMatrixMultiplyCore();

            BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuGemmLowpMatrixMultiplyCore);

            ~BICpuGemmLowpMatrixMultiplyCore();

            /**
             * 初始化内核的输入和输出。
             *
             * 支持的数据布局（Data Layouts）：
             * - NHWC
             * - NCHW
             *
             *  支持的数据类型配置：
             * |src0           |src1               |src2     |dst            |
             * |:--------------|:------------------|:--------|:--------------|
             * |QASYMM8        |QASYMM8            |S32      |QASYMM8        |
             * |QASYMM8        |QASYMM8_SIGNED     |S32      |QASYMM8        |
             * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
             * |QASYMM8        |QSYMM8             |S32      |QASYMM8        |
             * |QASYMM8        |QASYMM8            |S32      |S32            |
             * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |S32            |
             * |QASYMM8        |QSYMM8             |S32      |S32            |
             * |QASYMM8        |QASYMM8_SIGNED     |F32      |F32            |
             * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
             * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
             * |QASYMM8_SIGNED |QSYMM8             |S32      |QASYMM8_SIGNED |
             * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |S32            |
             * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |S32            |
             * |QASYMM8_SIGNED |QSYMM8             |S32      |S32            |
             * |QASYMM8_SIGNED |QASYMM8_SIGNED     |F32      |F32            |
             *
             * @note GEMM_LOWP: 低精度 GEMM 内核
             *  此内核执行以下计算：
             *
             *  1. 将矩阵 A 的值从 QASYMM8 转换为 int32，并将 a_offset 添加到每个值。
             *  2. 将矩阵 B 的值从 QASYMM8 转换为 int32，并将 b_offset 添加到每个值。
             *  3. 计算转换后的矩阵 A 和矩阵 B 的乘积，结果为 int32 类型。
             *
             * @note 如果 @p gemm_info.type == GEMMLowpOutputStageType::NONE，则 @p 输出类型为 S32。
             *       否则，输出类型为 QASYMM8/QASYMM8_SIGNED/F32。
             *
             *
             *
             * @param a 第一个输入张量（矩阵 A）的信息。支持的数据类型：QASYMM8/QASYMM8_SIGNED。
             * @param b 二个输入张量（矩阵 B）的信息。支持的数据类型：QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL。
             * @param c 第三个输入张量（矩阵 C）的信息。可以为 nullptr。支持的数据类型：S32/F32。
             * @param dst 输出张量的信息。支持的数据类型：S32/QASYMM8/QASYMM8_SIGNED/F32。
             * @param gemm_info （可选）指定矩阵 A 和/或矩阵 B 是否已被重排，以及矩阵 B 的重排是否仅在第一次运行时执行。
             */
            void configure(const BIITensorInfo *a,
                           const BIITensorInfo *b,
                           const BIITensorInfo *c,
                           BIITensorInfo *dst,
                           const GEMMInfo &gemm_info = GEMMInfo());

            static BIStatus validate(const BIITensorInfo *a,
                                     const BIITensorInfo *b,
                                     const BIITensorInfo *c,
                                     const BIITensorInfo *dst,
                                     const GEMMInfo &gemm_info = GEMMInfo());

            void run(BatmanInfer::BIITensorPack &tensors) override;

            void prepare(BatmanInfer::BIITensorPack &constants) override;

            experimental::BIMemoryRequirements workspace() const override;

            void update_quantization_parameters(const BIGEMMLowpOutputStageInfo &output_info,
                                                const BIQuantizationInfo &a,
                                                const BIQuantizationInfo &b,
                                                const bool is_prepared,
                                                const bool negated_offsets);

            /**
             * @brief 动态配置输入张量
             * @param a
             */
            void dynamic_configure(const BIITensorInfo *a) const;

        private:
            enum BIAuxTensorIdx {
                /* Slots 0 - 2 reserved for BICpuGemmAssemblyDispatch */
                VectorSumCol = 3,
                VectorSumRow,
                TmpA,
                TmpB,
                MMResultS32,
                SignedA,
                SignedOutput,
                Count
            };

            // 内核调度相关的对象
            // 负责处理 GEMM（通用矩阵乘法）操作的调度器，用于优化矩阵乘法的执行路径，可能使用汇编加速。
            std::unique_ptr<BICpuGemmAssemblyDispatch> _asm_glue;

            // 内核组件
            // 负责执行低精度（low precision）矩阵乘法（int8 或 uint8）的内核。
            std::unique_ptr<kernels::BICpuGemmLowpMatrixMultiplyKernel> _mm_kernel;

            // 负责对矩阵 A 进行 4x4 的交错重排（interleave），以提高矩阵乘法的内存访问效率。
            std::unique_ptr<kernels::BICpuGemmInterleave4x4Kernel> _mtx_a_reshape_kernel;

            // 负责对矩阵 B 进行转置重排（transpose），以适配矩阵乘法的计算模式。
            std::unique_ptr<kernels::BICpuGemmTranspose1xWKernel> _mtx_b_reshape_kernel;

            // 负责计算矩阵 A 的按行求和，用于偏移校正（offset contribution）。
            std::unique_ptr<kernels::BICpuGemmLowpMatrixAReductionKernel> _mtx_a_reduction_kernel;

            // 负责计算矩阵 B 的按列求和，用于偏移校正（offset contribution）。
            std::unique_ptr<kernels::BICpuGemmLowpMatrixBReductionKernel> _mtx_b_reduction_kernel;

            // 负责将矩阵 A 和 B 的零点偏移（offset contribution）应用到矩阵乘法结果中。
            std::unique_ptr<kernels::BICpuGemmLowpOffsetContributionKernel> _offset_contribution_kernel;

            // 负责在添加偏移贡献后执行量化输出阶段（output stage），将结果从 int32 转换为 int8 或 uint8。
            std::unique_ptr<kernels::BICpuGemmLowpOffsetContributionOutputStageKernel>
            _offset_contribution_output_stage_kernel;

            // 可选的激活函数对象（如 ReLU、ReLU6 等），用于在输出阶段对结果应用激活函数。
            std::unique_ptr<BICpuActivation> _activation_func;

            // 负责将矩阵 A 的量化数据从无符号（unsigned）转换为有符号（signed）格式。
            std::unique_ptr<kernels::BICpuConvertQuantizedSignednessKernel> _convert_to_signed_asymm;

            // 负责将量化结果从有符号（signed）转换为无符号（unsigned）格式。
            std::unique_ptr<kernels::BICpuConvertQuantizedSignednessKernel> _convert_from_signed_asymm;

            // Tensor信息
            // 存储矩阵B的每列求和结果，用于偏正校正
            BITensorInfo _vector_sum_col;

            // 存储矩阵A的每行求和结果，用于偏正校正
            BITensorInfo _vector_sum_row;

            // 存储矩阵 A 的临时重排结果。
            BITensorInfo _tmp_a;

            // 存储矩阵 B 的临时重排结果。
            BITensorInfo _tmp_b;

            // 存储矩阵乘法的 int32 结果（未量化前）。
            BITensorInfo _mm_result_s32;

            // 存储转换为有符号格式的矩阵 A。
            BITensorInfo _signed_a;

            // 存储转换为有符号格式的输出结果。
            BITensorInfo _signed_output;

            // 矩阵 A 的量化零点（offset）。
            int32_t _a_offset;

            // 矩阵 B 的量化零点（offset）。
            int32_t _b_offset;

            // 控制标志，  控制是否运行矢量-矩阵乘法。
            bool _run_vector_matrix_multiplication;

            // 控制是否使用汇编优化路径。
            bool _assembly_path;

            // 控制是否使用融合的汇编优化路径（可能包括偏移校正和量化）。
            bool _fused_assembly_path;

            // 控制是否仅在第一次运行时对矩阵 B 进行重排。
            bool _reshape_b_only_on_first_run;

            // 标志是否已经完成准备工作（如重排和偏移计算）。
            bool _is_prepared;

            // 控制是否将偏移校正和量化输出阶段融合为一个步骤。
            bool _fuse_output_stage;

            // 控制是否运行激活函数。
            bool _run_activation;

            // 控制是否需要切换量化数据的符号（signed/unsigned）。
            bool _flip_signedness;

            // 存储 GEMM（矩阵乘法）的配置信息，例如是否需要重排、是否需要偏移校正等。
            GEMMInfo _gemm_info;

            // 内存需求
            experimental::BIMemoryRequirements _aux_mem{};
        };
    }
}
