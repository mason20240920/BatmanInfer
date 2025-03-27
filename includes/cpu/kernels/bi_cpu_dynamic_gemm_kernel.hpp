//
// Created by Mason on 2025/3/26.
//

#pragma once

#include <data/core/bi_tensor_info.hpp>

#include <data/core/common/bi_core_common_macros.hpp>
#include <data/core/helpers/bi_memory_helpers.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>
#include <cpu/kernels/dynamic_gemm/heuristics/bi_cpu_dynamic_gemm_kernel_heuristics.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /**
             * @brief Arm(R) Neon (TM) kernel to perform dynamic GEMM
             */
            class BICpuDynamicGemmKernel final : public BIICpuKernel<BICpuDynamicGemmKernel> {
            public:
                BICpuDynamicGemmKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuDynamicGemmKernel);

                /** Initialise the kernel's input and output.
                 *
                 * @param[in]  a             First input tensor info (Matrix A or Vector A). Data type supported: F32
                 * @param[in]  b             Second input tensor info (Matrix B). Data type supported: same as @p a
                 * @param[in]  c             Third input tensor info (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a
                 * @param[out] d             Output tensor info. Data type supported: same as @p a
                 * @param[in]  alpha         Weight of the matrix product
                 * @param[in]  beta          Weight of matrix C
                 * @param[in]  base_aux_slot First slot to use for intermediate tensor allocations
                 * @param[in]  gemm_info     (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
                 *                           if the reshape of matrix B should happen only for the first run
                 */
                void configure(const BIITensorInfo *a,
                               const BIITensorInfo *b,
                               const BIITensorInfo *c,
                               BIITensorInfo *d,
                               float alpha,
                               float beta,
                               size_t base_aux_slot,
                               const GEMMInfo &gemm_info = GEMMInfo());

                /** Static function to check if given info will lead to a valid configuration of @ref BICpuDynamicGemmMatKernel.
                 *
                 * @note The input and output tensor must have the same dimensions
                 *
                 * Similar to @ref BICpuDynamicGemmKernel::configure()
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
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

                /** Return updated extra memory requirements for the selected ukernel,
                 * based on the tensors that will be used when running it.
                 */
                const experimental::BIMemoryRequirements &workspace(const BIITensorPack &tensors) const;

                /** Return the maximum number of allocations that may be required by the
                 * selected ukernel.
                 */
                static constexpr size_t max_workspace_count();

                /** Prepare the kernel for the run.
                 *
                 * Any actions the kernel needs to perform before the run should be
                 * done here. An example of such an action could be packing RHS.
                 *
                 * @param[in] tensors Tensors to operate on.
                 * @param[in] reuse_b Whether b-tensor from the last run should
                 *                    be reused. This for instance allows to skip
                 *                    unnecessary packing of b-tensor if it was not
                 *                    changed.
                 */
                void prepare(BIITensorPack &tensors, const bool reuse_b);

                /** Get the preferred dimension in which the scheduler splits the work into multiple jobs.
                 *
                 * @return The split dimension hint.
                 */
                size_t get_split_dimension_hint() const {
                    return _heuristics.scheduler_hint().split_dimension();
                }

            private:
                // 可能需要为所选内核分配的中间张量类型。
                enum AuxTensorIdx {
                    PackedRHS = 0,
                    Count
                };

            private:
                // 如果这是所选的微内核所需要的，请计算打包过的RHS的大小。
                size_t size_of_packed_rhs(size_t rows, size_t columns) const;

            private:
                heuristics::BICpuDynamicGemmKernelHeuristics _heuristics{};
                std::string _name{};
                size_t _base_aux_slot{};
                // `mutable` to be able to cache and return memory requirements from the
                // `workspace` method.
                mutable experimental::BIMemoryRequirements _aux_mem{Count};
            };

            constexpr size_t BICpuDynamicGemmKernel::max_workspace_count() {
                return Count;
            }
        }
    }
}
