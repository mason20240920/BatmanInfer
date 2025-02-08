//
// Created by Mason on 2025/2/8.
//

#include <runtime/neon/functions/bi_NEDequantizationLayer.h>

#include <data/core/bi_vlidate.hpp>
#include <cpu/operators/bi_cpu_dequantize.hpp>

namespace BatmanInfer {
    struct BINEDequantizationLayer::Impl {
        const BIITensor *src{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuDequantize> op{nullptr};
    };

    BINEDequantizationLayer::BINEDequantizationLayer() : _impl(std::make_unique<Impl>()) {
    }

    BINEDequantizationLayer::~BINEDequantizationLayer() = default;

    void BINEDequantizationLayer::configure(const BIITensor *input, BIITensor *output) {
        _impl->src = input;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuDequantize>();
        _impl->op->configure(input->info(), output->info());
    }

    BIStatus BINEDequantizationLayer::validate(const BIITensorInfo *input, const BIITensorInfo *output) {
        return cpu::BICpuDequantize::validate(input, output);
    }

    void BINEDequantizationLayer::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC, _impl->src);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }
}