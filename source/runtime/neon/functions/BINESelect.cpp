//
// Created by Mason on 2025/4/3.
//

#include <runtime/neon/functions/BINESelect.hpp>

#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/BINESelectKernel.hpp>

namespace BatmanInfer {
    void BINESelect::configure(const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output) {
        BI_COMPUTE_LOG_PARAMS(c, x, y, output);
        auto k = std::make_unique<BINESelectKernel>();
        k->configure(c, x, y, output);
        _kernel = std::move(k);
    }

    BIStatus BINESelect::validate(const BIITensorInfo *c, const BIITensorInfo *x, const BIITensorInfo *y,
                                  const BIITensorInfo *output) {
        BI_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(c, x, y, output);
        return BINESelectKernel::validate(c, x, y, output);
    }
}
