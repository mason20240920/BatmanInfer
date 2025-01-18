//
// Created by Mason on 2025/1/18.
//

#pragma once

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/bi_i_cpu_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            /** Kernel to perform a copy between two tensors */
            class BICpuCopyKernel : public BIICpuKernel<BICpuCopyKernel> {
            public:
                BICpuCopyKernel() = default;

                BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuCopyKernel);

                /** Configure kernel for a given list of arguments
                 *
                 * @param[in]  src     Source tensor. Data types supported: All
                 * @param[out] dst     Destination tensor. Data types supported: same as @p src.
                 * @param[in]  padding (Optional) Padding to be applied to the input tensor
                 */
                void
                configure(const BIITensorInfo *src, BIITensorInfo *dst, const PaddingList &padding = PaddingList());

                /** Static function to check if given info will lead to a valid configuration
                 *
                 * Similar to @ref CpuCopyKernel::configure()
                 *
                 * @return a status
                 */
                static BIStatus
                validate(const BIITensorInfo *src, const BIITensorInfo *dst,
                         const PaddingList &padding = PaddingList());

                // Inherited methods overridden:
                void run_op(BIITensorPack &tensors, const BIWindow &window, const ThreadInfo &info) override;

                const char *name() const override;

            private:
                PaddingList _padding{};
            };
        } // namespace kernels
    } // namespace cpu
}