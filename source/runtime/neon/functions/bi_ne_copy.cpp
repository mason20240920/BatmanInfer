//
// Created by Mason on 2025/1/23.
//

#include <runtime/neon/functions/bi_ne_copy.hpp>

#include <data/core/bi_vlidate.hpp>

#include <cpu/operators/bi_cpu_copy.hpp>

namespace BatmanInfer {
    struct BINECopy::Impl {
        const BIITensor *src{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuCopy> op{nullptr};
    };

    BINECopy::BINECopy() : _impl(std::make_unique<Impl>()) {

    }

    BINECopy::BINECopy(BINECopy &&) = default;

    BINECopy &BINECopy::operator=(BINECopy &&) = default;

    BINECopy::~BINECopy() = default;

    void BINECopy::configure(BatmanInfer::BIITensor *input,
                             BatmanInfer::BIITensor *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);

        _impl->src = input;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuCopy>();
        _impl->op->configure(input->info(), output->info());
    }

    BIStatus BINECopy::validate(const BatmanInfer::BIITensorInfo *input,
                                const BatmanInfer::BIITensorInfo *output) {
        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_RETURN_ON_ERROR(cpu::BICpuCopy::validate(input, output));

        return BIStatus{};
    }

    void BINECopy::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC, _impl->src);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }
}