//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_BI_CPU_GEMM_HPP
#define BATMANINFER_BI_CPU_GEMM_HPP

#include <function_info/bi_activationLayerInfo.h>
#include <data/core/bi_i_tensor_pack.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <function_info/bi_GEMMInfo.h>

#include <cpu/kernels/bi_cpu_gemm_inter_leave_4x4_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_matrix_addition_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_matrix_multiply_kernel.hpp>
#include <cpu/kernels/bi_cpu_gemm_transpose_1xw_kernel.hpp>
#include <cpu/operators/bi_cpu_activation.hpp>
#include <cpu/operators/bi_cpu_add.hpp>
#include <cpu/operators/cpu_transpose.hpp>
#include <cpu/operators/internal/cpu_gemm_assembly_dispatch.hpp>

#include <cpu/bi_i_cpu_operator.hpp>

namespace BatmanInfer {
    namespace cpu {
        /**
         * 执行 GEMM（广义矩阵乘法）的基本函数。
         * 此函数会调用以下内核：
         *
         * 如果有可用的优化汇编：
         * 1. @ref cpu::BICpuGemmAssemblyDispatch
         * 2. @ref cpu::BICpuActivation （如果 alpha != 1.0）
         *
         * 否则：
         * 1. @ref cpu::kernels::CpuGemmInterleave4x4Kernel （如果输出张量是一个矩阵）
         * 2. @ref cpu::kernels::CpuGemmTranspose1xWKernel （如果输出张量是一个矩阵）
         * 3. @ref cpu::kernels::CpuGemmMatrixMultiplyKernel
         *
         * 在两种情况下：
         * 1. @ref cpu::kernels::CpuGemmMatrixAdditionKernel （如果 c != nullptr 且 beta != 0.0 且未经过一次 reshape）
         * 否则：
         *  1. @ref cpu::CpuAdd （如果 c != nullptr 且经过一次 reshape 且没有可用的优化汇编）
         *
         *  1. @ref cpu::CpuActivation （如果在 GEMMInfo 中指定了激活函数）
         *
         */
        class BICpuGemm : public BIICpuOperator {
        public:
            BICpuGemm() = default;

            ~BICpuGemm() = default;

            /**
             * 配置操作符以处理给定的参数列表。
             *
             * 有效的数据布局：
             * - 全部
             *
             * 有效的数据类型配置：
             * | src0        | src1       | src2      | dst            |
             * |:------------|:-----------|:----------|:---------------|
             * | F32         | F32        | F32       | F32            |
             * | F16         | F16        | F16       | F16            |
             * | BFLOAT16    | BFLOAT16   | BFLOAT16  | FP32           |
             *
             * @note GEMM: 广义矩阵乘法 - [alpha * A * B + beta * C]。
             * @note GEMM: 张量 a, b, c, d 必须具有相同的数据类型。调用此函数时，不应混用不同的数据类型。
             *
             * @note 对于批量 GEMM，仅支持 RHS（右手边张量）的秩小于 LHS（左手边张量）的广播情况，不支持反向情况。
             *
             * @param a 第一个输入张量信息（矩阵 A 或向量 A）。支持的数据类型：BFLOAT16/F16/F32
             * @param b 第二个输入张量信息（矩阵 B）。支持的数据类型：与 @p a 相同
             * @param c 第三个输入张量信息（矩阵 C）。如果只需要计算 @p a 和 @p b 的乘积，可以为 nullptr。支持的数据类型：与 @p a 相同
             * @param d 输出张量信息。支持的数据类型：与 @p a 相同
             * @param alpha 矩阵乘积的权重
             * @param beta 矩阵 C 的权重
             * @param gemm_info (可选）指定矩阵 A 和/或矩阵 B 是否已被重排， 以及矩阵 B 的重排是否仅在第一次运行时发生
             */
            void configure(const BIITensorInfo *a,
                           const BIITensorInfo *b,
                           const BIITensorInfo *c,
                           BIITensorInfo *d,
                           float alpha,
                           float beta,
                           const GEMMInfo &gemm_info = GEMMInfo());

            void dynamic_configure(const BIITensorInfo *a,
                                   const BIITensorInfo *b,
                                   BIITensorInfo *d);

            static BIStatus validate(const BIITensorInfo *a,
                                     const BIITensorInfo *b,
                                     const BIITensorInfo *c,
                                     const BIITensorInfo *d,
                                     float alpha,
                                     float beta,
                                     const GEMMInfo &gemm_info = GEMMInfo());

            /**
             * 指示是否存在可用于处理给定参数的优化汇编实现
             *
             * 此方法的用途与 @ref NEGEMMConvolutionLayer::has_opt_impl 相同，
             *
             * 唯一的区别是需要通过参数 gemm_info 传递 arm_compute::WeightFormat 的值
             * @param weight_format
             * @param a
             * @param b
             * @param c
             * @param d
             * @param gemm_info
             * @return
             */
            static BIStatus has_opt_impl(BatmanInfer::BIWeightFormat &weight_format,
                                         const BIITensorInfo *a,
                                         const BIITensorInfo *b,
                                         const BIITensorInfo *c,
                                         const BIITensorInfo *d,
                                         const GEMMInfo &gemm_info = GEMMInfo());

            void run(BatmanInfer::BIITensorPack &tensors) override;

            void prepare(BatmanInfer::BIITensorPack &constants) override;

            experimental::BIMemoryRequirements workspace() const override;


            /**
             *  指示卷积是否以可变权重模式执行
             *  当 ACL 以可变权重模式执行卷积时，它不会对权重张量进行任何处理。
             *  相反，它直接使用用户提供的数据
             * @return
             */
            bool isVarWeightsKernel() const;

        private:
            enum AuxTensorIdx {
                /* Slots 0 - 2 reserved for CpuGemmAssemblyDispatch */
                InterleavedLHS = 3, // 标记左侧矩阵（LHS）的交错存储版本
                PreTransposedRHS, // 标记右侧矩阵（RHS）的预转置版本
                Transposed1xWRHS, // 标记右侧矩阵（RHS）的 1xW 转置版本
                TempResult, // 标记临时存储的计算结果
                Count // 表示枚举值的总数（通常用于循环或分配数组大小）
            };

            std::unique_ptr<kernels::BICpuGemmInterleave4x4Kernel> _interleave_kernel{nullptr};
            std::unique_ptr<BICpuTranspose> _pretranspose_b_func{nullptr};
            std::unique_ptr<kernels::BICpuGemmTranspose1xWKernel> _transpose1xW_b_kernel{nullptr};
            std::unique_ptr<kernels::BICpuGemmMatrixMultiplyKernel> _mm_kernel{nullptr};
            std::unique_ptr<BICpuGemmAssemblyDispatch> _asm_glue{nullptr};
            std::unique_ptr<kernels::BICpuGemmMatrixAdditionKernel> _ma_kernel{nullptr};
            std::unique_ptr<BICpuActivation> _alpha_scale_func{nullptr};
            std::unique_ptr<BICpuAdd> _add_bias{nullptr};
            std::unique_ptr<BICpuActivation> _activation_func{nullptr};

            BITensorInfo _tmp_a{};
            BITensorInfo _pretransposed_b{};
            BITensorInfo _tmp_b{};
            BITensorInfo _tmp_d{};

            bool _run_vector_matrix_multiplication{false};
            bool _run_interleave_transpose{
                true
            }; /**< If we run CpuGemmInterleave4x4Kernel on lhs and CpuGemmTranspose1xWKernel on rhs */
            bool _run_alpha_scale{false};
            bool _run_addition{false};
            bool _run_bias_addition{false};
            bool _run_activation{false};
            bool _reshape_b_only_on_first_run{false};
            bool _is_prepared{false};

            experimental::BIMemoryRequirements _aux_mem{Count};
        };
    }
}

#endif //BATMANINFER_BI_CPU_GEMM_HPP
