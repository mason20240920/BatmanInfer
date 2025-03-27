//
// Created by Mason on 2025/3/26.
//

#pragma once

#include <data/core/bi_i_tensor_info.hpp>

#include <cpu/bi_i_cpu_operator.hpp>
#include <cpu/kernels/bi_cpu_dynamic_gemm_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        /**
         * @brief Basic function to execute dynamic GEMM. This function calls the following kernels:
         *  - #  @ref: cpu::kernels::BICpuDynamicGemmKernel
         */
        class BICpuDynamicGemm : public BIICpuOperator {
        public:
            /** Default constructor */
            BICpuDynamicGemm() = default;

            /** Default destructor */
            ~BICpuDynamicGemm() = default;

            /** Configure operator for a given list of arguments
             *
             * @note GEMM: General Matrix Multiply - [alpha * A * B + beta * C].
             * @note GEMM: The tensors a, b, c, d must have the same data type. You should not mix data types when calling this function.
             *
             * @param[in]  a         First input tensor info (Matrix A or Vector A). Data type supported: F32
             * @param[in]  b         Second input tensor info (Matrix B). Data type supported: same as @p a
             * @param[in]  c         Third input tensor info (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a
             * @param[out] d         Output tensor info. Data type supported: same as @p a
             * @param[in]  alpha     Weight of the matrix product
             * @param[in]  beta      Weight of matrix C
             * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
             *                       if the reshape of matrix B should happen only for the first run
             */
            void configure(const BIITensorInfo *a,
                           const BIITensorInfo *b,
                           const BIITensorInfo *c,
                           BIITensorInfo *d,
                           float alpha,
                           float beta,
                           const GEMMInfo &gemm_info = GEMMInfo());

            /** Static function to check if given info will lead to a valid configuration of @ref CpuDynamicGemm.
             *
             * Similar to @ref CpuDynamicGemm::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *a,
                                     const BIITensorInfo *b,
                                     const BIITensorInfo *c,
                                     const BIITensorInfo *d,
                                     float alpha,
                                     float beta,
                                     const GEMMInfo &gemm_info = GEMMInfo());

            // Inherited methods overridden:
            void run(BIITensorPack &tensors) override;

            const experimental::BIMemoryRequirements &workspace_dynamic(const BIITensorPack &tensors) const override;

        private:
            enum AuxTensorIdx {
                Count
            };

            std::unique_ptr<kernels::BICpuDynamicGemmKernel> _kernel{nullptr};
            mutable experimental::BIMemoryRequirements _aux_mem{};
            bool _reshape_b_and_c_only_on_first_run{false};
            bool _reuse_b{false};
        };
    }
}
