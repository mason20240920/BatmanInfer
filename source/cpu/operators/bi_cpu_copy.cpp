//
// Created by Mason on 2025/1/23.
//

#include <cpu/operators/bi_cpu_copy.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_cpu_copy_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        void BICpuCopy::configure(const BatmanInfer::BIITensorInfo *src,
                                  BatmanInfer::BIITensorInfo *dst) {
            BI_COMPUTE_LOG_PARAMS(src, dst);
            auto k = std::make_unique<kernels::BICpuCopyKernel>();
            k->configure(src, dst);
            _kernel = std::move(k);
        }

        void BICpuCopy::dynamic_configure(BIITensorInfo *dst) {
            auto k = reinterpret_cast<kernels::BICpuCopyKernel *>(_kernel.get());
            k->dynamic_configure(dst);
        }


        BIStatus BICpuCopy::validate(const BatmanInfer::BIITensorInfo *src,
                                     const BatmanInfer::BIITensorInfo *dst) {
            return kernels::BICpuCopyKernel::validate(src, dst);
        }
    }
} // namespace BatmanInfer
