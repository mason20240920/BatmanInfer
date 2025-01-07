//
// Created by Mason on 2025/1/7.
//

#include <cpu/operators/cpu_transpose.hpp>
#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_cpu_transpose_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        void BICpuTranspose::configure(const BIITensorInfo *src, BIITensorInfo *dst) {
            BI_COMPUTE_LOG_PARAMS(src, dst);
            auto k = std::make_unique<kernels::BICpuTransposeKernel>();
            k->configure(src, dst);
            _kernel = std::move(k);
        }

        BIStatus
        BICpuTranspose::validate(const BIITensorInfo *src, const BIITensorInfo *dst) {
            return kernels::BICpuTransposeKernel::validate(src, dst);
        }
    }
}