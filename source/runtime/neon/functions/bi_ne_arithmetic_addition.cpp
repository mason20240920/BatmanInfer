//
// Created by Mason on 2025/1/23.
//

#include <runtime/neon/functions/bi_NEArithmeticAddition.h>

#include <data/core/bi_vlidate.hpp>

#include <cpu/operators/bi_cpu_add.hpp>

namespace BatmanInfer {
    struct BINEArithmeticAddition::Impl {
        const BIITensor *src_0{nullptr};
        const BIITensor *src_1{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuAdd> op{nullptr};
    };

    BINEArithmeticAddition::BINEArithmeticAddition() : _impl(std::make_unique<Impl>()) {}

    BINEArithmeticAddition::BINEArithmeticAddition(BINEArithmeticAddition &&) = default;

    BINEArithmeticAddition &BINEArithmeticAddition::operator=(BINEArithmeticAddition &&) = default;

    BINEArithmeticAddition::~BINEArithmeticAddition() = default;

    BIStatus
    BINEArithmeticAddition::validate(const BatmanInfer::BIITensorInfo *input1,
                                     const BatmanInfer::BIITensorInfo *input2,
                                     const BatmanInfer::BIITensorInfo *output,
                                     BatmanInfer::BIConvertPolicy policy,
                                     const BatmanInfer::BIActivationLayerInfo &act_info) {
        return cpu::BICpuAdd::validate(input1, input2, output, policy, act_info);
    }

    void BINEArithmeticAddition::configure(const BatmanInfer::BIITensor *input1,
                                           const BatmanInfer::BIITensor *input2,
                                           BatmanInfer::BIITensor *output,
                                           BatmanInfer::BIConvertPolicy policy,
                                           const BatmanInfer::BIActivationLayerInfo &act_info) {
        _impl->src_0 = input1;
        _impl->src_1 = input2;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuAdd>();
        _impl->op->configure(_impl->src_0->info(), _impl->src_1->info(), _impl->dst->info(), policy, act_info);
    }

    void BINEArithmeticAddition::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC_0, _impl->src_0);
        pack.add_tensor(BITensorType::ACL_SRC_1, _impl->src_1);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }
}