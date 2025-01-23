//
// Created by Mason on 2025/1/23.
//

#include <runtime/neon/functions/bi_NEActivationLayer.h>

#include <data/core/bi_vlidate.hpp>

#include <cpu/operators/bi_cpu_activation.hpp>

namespace BatmanInfer {
    struct BINEActivationLayer::Impl {
        const BIITensor *src{nullptr};
        BIITensor *dst{nullptr};
        BIIRuntimeContext *ctx{nullptr};
        std::unique_ptr<cpu::BICpuActivation> op{nullptr};
    };

    BINEActivationLayer::BINEActivationLayer(BatmanInfer::BIIRuntimeContext *ctx) : _impl(std::make_unique<Impl>()) {
        _impl->ctx = ctx;
    }

    BINEActivationLayer::BINEActivationLayer(BatmanInfer::BINEActivationLayer &&) = default;

    BINEActivationLayer &BINEActivationLayer::operator=(BINEActivationLayer &&) = default;

    BINEActivationLayer::~BINEActivationLayer() = default;

    void BINEActivationLayer::configure(BatmanInfer::BIITensor *input,
                                        BatmanInfer::BIITensor *output,
                                        BatmanInfer::BIActivationLayerInfo activation_info) {
        _impl->src = input;
        _impl->dst = output == nullptr ? input : output;

        BI_COMPUTE_ERROR_ON_NULLPTR(_impl->src, _impl->dst);

        _impl->op = std::make_unique<cpu::BICpuActivation>();
        _impl->op->configure(_impl->src->info(), _impl->dst->info(), activation_info);
    }

    BIStatus
    BINEActivationLayer::validate(const BatmanInfer::BIITensorInfo *input,
                                  const BatmanInfer::BIITensorInfo *output,
                                  const BatmanInfer::BIActivationLayerInfo &act_info) {
        return cpu::BICpuActivation::validate(input, output, act_info);
    }

    void BINEActivationLayer::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC, _impl->src);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }

}