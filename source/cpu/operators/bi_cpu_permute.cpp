//
// Created by Mason on 2025/1/18.
//

#include <cpu/operators/bi_cpu_permute.hpp>

#include <data/core/core_types.hpp>
#include <data/core/bi_error.h>
#include <data/core/bi_i_tensor_info.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_cpu_copy_kernel.hpp>
#include <cpu/kernels/bi_cpu_permute_kernel.hpp>
#include <cpu/kernels/bi_cpu_transpose_kernel.hpp>

#include <algorithm>
#include <array>
#include <memory>

namespace BatmanInfer {
    namespace cpu {
        namespace {
            // Handle "No-op" cases
            bool prefer_copy(const PermutationVector &v) {
                static const std::array<PermutationVector, 6> permutations = {
                    {
                        PermutationVector(0U),
                        PermutationVector(0U, 1U),
                        PermutationVector(0U, 1U, 2U),
                        PermutationVector(0U, 1U, 2U, 3U),
                        PermutationVector(0U, 1U, 2U, 3U,
                                          4U),
                        PermutationVector(0U, 1U, 2U, 3U,
                                          4U, 5U),
                    }
                };

                return std::find(permutations.begin(), permutations.end(), v) != permutations.end();
            }

            // Transpose kernel is optimized for permuting the first two dimensions of a tensor
            bool prefer_transpose(const PermutationVector &v) {
                static const std::array<PermutationVector, 5> permutations = {
                    {
                        PermutationVector(1U, 0U),
                        PermutationVector(1U, 0U, 2U),
                        PermutationVector(1U, 0U, 2U, 3U),
                        PermutationVector(1U, 0U, 2U, 3U,
                                          4U),
                        PermutationVector(1U, 0U, 2U, 3U,
                                          4U, 5U),
                    }
                };

                return std::find(permutations.begin(), permutations.end(), v) != permutations.end();
            }
        } // namespace

        void BICpuPermute::configure(const BIITensorInfo *src, BIITensorInfo *dst, const PermutationVector &perm) {
            BI_COMPUTE_LOG_PARAMS(src, dst, perm);

            if (prefer_copy(perm)) {
                auto k = std::make_unique<kernels::BICpuCopyKernel>();
                k->configure(src, dst);
                _kernel = std::move(k);
            } else if (prefer_transpose(perm)) {
                auto k = std::make_unique<kernels::BICpuTransposeKernel>();
                k->configure(src, dst);
                _kernel = std::move(k);
            } else {
                auto k = std::make_unique<kernels::BICpuPermuteKernel>();
                k->configure(src, dst, perm);
                _kernel = std::move(k);
            }
        }

        void BICpuPermute::dynamic_configure(const BIITensorInfo *src, BIITensorInfo *dst) {
            // 方法1：使用 dynamic_cast
            if (auto *copy_kernel = dynamic_cast<kernels::BICpuCopyKernel *>(_kernel.get())) {
                auto k = reinterpret_cast<kernels::BICpuCopyKernel *>(_kernel.get());
                k->dynamic_configure(dst);
            } else if (auto *transpose_kernel = dynamic_cast<kernels::BICpuTransposeKernel *>(_kernel.get())) {
                auto k = reinterpret_cast<kernels::BICpuTransposeKernel *>(_kernel.get());
                k->dynamic_configure(src, dst);
            } else if (auto *permute_kernel = dynamic_cast<kernels::BICpuPermuteKernel *>(_kernel.get())) {
                auto k = reinterpret_cast<kernels::BICpuPermuteKernel *>(_kernel.get());
                k->dynamic_configure(src);
            }
        }


        BIStatus
        BICpuPermute::validate(const BIITensorInfo *src, const BIITensorInfo *dst, const PermutationVector &perm) {
            if (prefer_copy(perm)) {
                return kernels::BICpuCopyKernel::validate(src, dst);
            }

            if (prefer_transpose(perm)) {
                return kernels::BICpuTransposeKernel::validate(src, dst);
            }

            return kernels::BICpuPermuteKernel::validate(src, dst, perm);
        }
    } // namespace cpu
}
