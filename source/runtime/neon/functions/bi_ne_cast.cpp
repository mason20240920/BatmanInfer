//
// Created by holynova on 25-4-10.
//

#include "runtime/neon/functions/bi_ne_cast.h"

#include "data/core/bi_vlidate.hpp"
#include "common/utils/bi_log.hpp"
#include "cpu/operators/bi_cpu_cast.h"

namespace BatmanInfer {

    struct BINECast::Impl
    {
        const BIITensor                *src{nullptr};
        BIITensor                      *dst{nullptr};
        std::unique_ptr<cpu::BICpuCast> op{nullptr};
    };

    BINECast::BINECast() : _impl(std::make_unique<Impl>())
    {
    }

    BINECast::BINECast(BINECast &&) noexcept = default;
    BINECast &BINECast::operator=(BINECast &&) noexcept = default;
    BINECast::~BINECast() = default;

    void BINECast::configure(BIITensor *input, BIITensor *output, BIConvertPolicy policy)
    {
        _impl->src = input;
        _impl->dst = output;

        BI_COMPUTE_ERROR_ON_NULLPTR(_impl->src, _impl->dst);
        BI_COMPUTE_LOG_PARAMS(input, output, policy);
        _impl->op = std::make_unique<cpu::BICpuCast>();
        _impl->op->configure(_impl->src->info(), _impl->dst->info(), policy);
    }

    BIStatus BINECast::validate(const BIITensorInfo *input, const BIITensorInfo *output, BIConvertPolicy policy)
    {
        BI_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(input, output);
        return cpu::BICpuCast::validate(input, output, policy);
    }

    void BINECast::run()
    {
        BIITensorPack pack = {{ACL_SRC, _impl->src}, {ACL_DST, _impl->dst}};
        _impl->op->run(pack);
    }

} // namespace BatmanInfer
