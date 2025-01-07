//
// Created by Mason on 2025/1/7.
//

#include <runtime/neon/functions/bi_ne_transpose.hpp>

#include <data/core/bi_vlidate.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/operators/cpu_transpose.hpp>

namespace BatmanInfer {
    struct BINETranspose::Impl {
        const BIITensor *src{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuTranspose> op{nullptr};
    };

    BINETranspose::BINETranspose() : _impl(std::make_unique<Impl>()) {

    }

    BINETranspose::~BINETranspose() = default;

    void BINETranspose::configure(const BIITensor *input, BIITensor *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_LOG_PARAMS(input, output);

        _impl->src = input;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuTranspose>();
        _impl->op->configure(input->info(), output->info());
    }

    BIStatus
    BINETranspose::validate(const BIITensorInfo *input, const BIITensorInfo *output) {
        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_RETURN_ON_ERROR(cpu::BICpuTranspose::validate(input, output));
        return BIStatus{};
    }

    void BINETranspose::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC, _impl->src);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }
}