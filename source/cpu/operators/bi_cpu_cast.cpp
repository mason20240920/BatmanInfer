//
// Created by holynova on 25-4-9.
//

#include "cpu/operators/bi_cpu_cast.h"

#include "common/utils/bi_log.hpp"
#include "cpu/kernels/bi_cpu_cast_kernel.h"

namespace BatmanInfer {

namespace cpu {

    void BICpuCast::configure(const BIITensorInfo *src, BIITensorInfo *dst, BIConvertPolicy policy)
    {
        BI_COMPUTE_LOG_PARAMS(src, dst, policy);
        auto k = std::make_unique<kernels::BICpuCastKernel>();
        k->configure(src, dst, policy);
        _kernel = std::move(k);
    }

    BIStatus BICpuCast::validate(const BIITensorInfo *src, const BIITensorInfo *dst, BIConvertPolicy policy)
    {
        return kernels::BICpuCastKernel::validate(src, dst, policy);
    }

} // namespace cpu

} // namespace BatmanInfer
