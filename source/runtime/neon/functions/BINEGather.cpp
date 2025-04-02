//
// Created by Mason on 2025/4/1.
//

#include <runtime/neon/functions/BINEGather.hpp>

#include <data/core/bi_vlidate.hpp>

#include <common/utils/bi_log.hpp>
#include <data/core/neon/kernels/BINEGatherKernel.hpp>

#include <utility>

namespace BatmanInfer {
    void BINEGather::configure(const BIITensor *input,
                               const BIITensor *indices,
                               BIITensor *output,
                               int axis) {
        BI_COMPUTE_LOG_PARAMS(input, indices, output, axis);
        auto k = std::make_unique<BINEGatherKernel>();
        k->configure(input, indices, output, axis);
        _kernel = std::move(k);
    }

    BIStatus BINEGather::validate(const BIITensorInfo *input, const BIITensorInfo *indices, const BIITensorInfo *output,
                                  int axis) {
        BI_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(input, indices, output);
        return BINEGatherKernel::validate(input, indices, output, axis);
    }

    void BINEGather::dynamic_configure(const BIITensor *indices, BIITensor *output) const {
        if (auto derived_kernel = static_cast<BINEGatherKernel *>(_kernel.get())) {
            derived_kernel->dynamic_configure(indices, output);
        }
    }
}

