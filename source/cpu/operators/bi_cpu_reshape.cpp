//
// Created by Mason on 2025/1/8.
//

#include <cpu/operators/bi_cpu_reshape.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_cpu_reshape_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        void BICpuReshape::configure(const BIITensorInfo *src, BIITensorInfo *dst) {
            BI_COMPUTE_LOG_PARAMS(src, dst);
            auto k = std::make_unique<kernels::BICpuReshapeKernel>();
            k->configure(src, dst);
            _kernel = std::move(k);
        }

        BIStatus BICpuReshape::validate(const BIITensorInfo *src, const BIITensorInfo *dst) {
            return kernels::BICpuReshapeKernel::validate(src, dst);
        }

        void BICpuReshape::run(BIITensorPack &tensors) {
            BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
            if (!_is_prepared) {
                static_cast<kernels::BICpuReshapeKernel *>(_kernel.get())->prepare(tensors);
                _is_prepared = true;
            }

            BIICpuOperator::run(tensors);
        }
    }
}