//
// Created by Mason on 2025/1/18.
//

#pragma once

#include <data/core/bi_tensor_info.hpp>
#include <cpu/bi_i_cpu_operator.hpp>

namespace BatmanInfer {
    namespace cpu {
        /** Basic function to run @ref kernels::CpuPermuteKernel */
        class BICpuPermute : public BIICpuOperator {
        public:
            /** Configure operator for a given list of arguments
             *
             * @note Arbitrary permutation vectors are supported with rank not greater than 4
             *
             * @param[in]  src  Source tensor to permute. Data types supported: All
             * @param[out] dst  Destination tensor. Data types supported: Same as @p src
             * @param[in]  perm Permutation vector
             */
            void configure(const BIITensorInfo *src, BIITensorInfo *dst, const PermutationVector &perm);

            void dynamic_configure(const BIITensorInfo *src, BIITensorInfo *dst);

            /** Static function to check if given info will lead to a valid configuration
             *
             * Similar to @ref CpuPermute::configure()
             *
             * @return a status
             */
            static BIStatus validate(const BIITensorInfo *src, const BIITensorInfo *dst, const PermutationVector &perm);
        };
    } // namespace cpu
}
