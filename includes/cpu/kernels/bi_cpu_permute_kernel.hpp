//
// Created by Mason on 2025/1/18.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /** Kernel to perform tensor permutation given a permutation vector */
            class BICpuPermuteKernel : public BIICpuKernel<BICpuPermuteKernel> {
            public:
                BICpuPermuteKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuPermuteKernel);

                /** Configure kernel for a given list of arguments
                 *
                 * @note Arbitrary permutation vectors are supported with rank not greater than 4
                 *
                 * @param[in]  src  Source tensor to permute. Data types supported: All
                 * @param[out] dst  Destination tensor. Data types supported: Same as @p src
                 * @param[in]  perm Permutation vector
                 */
                void configure(const BIITensorInfo *src, BIITensorInfo *dst, const PermutationVector &perm);

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to @ref CpuPermuteKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus
                validate(const BIITensorInfo *src, const BIITensorInfo *dst, const PermutationVector &perm);

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                void dynamic_configure(const BIITensorInfo *src);

                const char *name() const override;

            private:
                PermutationVector _perm{};
            };
        }
    }
}
