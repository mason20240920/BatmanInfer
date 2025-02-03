//
// Created by Mason on 2025/2/3.
//

#include <runtime/neon/functions/bi_ne_permute.h>

#include <data/core/bi_vlidate.hpp>

#include <cpu/operators/bi_cpu_permute.hpp>

namespace BatmanInfer {
    struct BINEPermute::Impl {
        const BIITensor *src{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuPermute> op{nullptr};
    };

    BINEPermute::BINEPermute() : _impl(std::make_unique<Impl>()) {

    }

    BINEPermute::~BINEPermute() = default;

    void BINEPermute::configure(const BatmanInfer::BIITensor *input,
                                BatmanInfer::BIITensor *output,
                                const BatmanInfer::PermutationVector &perm) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);

        _impl->src = input;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuPermute>();
        _impl->op->configure(input->info(), output->info(), perm);
    }

    BIStatus BINEPermute::validate(const BatmanInfer::BIITensorInfo *input,
                                   const BatmanInfer::BIITensorInfo *output,
                                   const BatmanInfer::PermutationVector &perm) {
        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_RETURN_ON_ERROR(cpu::BICpuPermute::validate(input, output, perm));

        return BIStatus{};
    }

    void BINEPermute::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC, _impl->src);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }
}