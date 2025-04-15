//
// Created by Mason on 2025/1/16.
//

#pragma once

#include <data/core/bi_i_tensor_info.hpp>

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_operator.hpp>
#include <cpu/kernels/bi_cpu_transpose_kernel.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <cpu/operators/internal/cpu_gemm_assembly_dispatch.hpp>

namespace BatmanInfer {
    // 前向声明
    class BIMatMulInfo;

    class BICpuMatMulSettings;

    namespace cpu {
        /**
         * 执行矩阵乘法运算的函数。该函数调用以下函数/内核：
         *
         * 如果对于输入的lhs或rhs（或两者）启用了伴随(adjoint/adj)标志：
         * - # @ref cpu::kernels::BICpuTransposeKernel
         * 然后使用:
         * - # @ref cpu::BICpuGemmAssemblyDispatch
         */
        class BICpuMatMul : public BIICpuOperator {
        public:
            // 构造函数
            BICpuMatMul();

            // 析构函数
            ~BICpuMatMul() = default;

            BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuMatMul);

            /**
             * 用给定的参数配置算子
             * @param lhs
             * @param rhs
             * @param dst
             * @param info
             * @param settings
             * @param act_info
             */
            void configure(BIITensorInfo *lhs,
                           BIITensorInfo *rhs,
                           BIITensorInfo *dst,
                           const BIMatMulInfo &info,
                           const BICpuMatMulSettings &settings,
                           const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

            void dynamic_configure(BIITensorInfo *lhs,
                                   BIITensorInfo *rhs,
                                   BIITensorInfo *dst);

            static BIStatus validate(const BIITensorInfo *lhs,
                                     const BIITensorInfo *rhs,
                                     const BIITensorInfo *dst,
                                     const BIMatMulInfo &info,
                                     const BICpuMatMulSettings &settings,
                                     const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

            void run(BatmanInfer::BIITensorPack &tensors) override;

            experimental::BIMemoryRequirements workspace() const override;

            const experimental::BIMemoryRequirements &workspace_dynamic(const BIITensorPack &tensors) const override;

        private:
            enum BIInternalTensorIdx {
                /* Slots 0 - 2 reserved for CpuGemmAssemblyDispatch */
                TransposeLHS = 3,
                TransposeRHS,
                Count
            };

            // 定义matmul需要的unique pointers
            std::unique_ptr<kernels::BICpuTransposeKernel> _transpose_kernel_lhs{nullptr};
            std::unique_ptr<kernels::BICpuTransposeKernel> _transpose_kernel_rhs{nullptr};
            std::unique_ptr<BICpuGemmAssemblyDispatch> _asm_glue{nullptr};

            // 辅助内存中存储的张量的TensorInfo
            BITensorInfo _lhs_transposed{};
            BITensorInfo _rhs_transposed{};

            // 重塑张量和折叠维度之前的原始张量形状
            BITensorShape _original_lhs_shape{};
            BITensorShape _original_rhs_shape{};
            BITensorShape _original_dst_shape{};

            // Note : adj_lhs means the same as transposing lhs
            bool _adj_lhs{false};
            bool _adj_rhs{false};
            bool _fast_math{false};
            BIAsmGemmInfo _gemm_info{};
            mutable experimental::BIMemoryRequirements _aux_mem{Count};
        };
    }
}
