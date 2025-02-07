//
// Created by Mason on 2025/2/7.
//

#include <runtime/neon/functions/bi_NEQuantizationLayer.h>
#include <data/core/bi_vlidate.hpp>
#include <runtime/bi_tensor.hpp>
#include <cpu/operators/bi_cpu_quantize.hpp>


namespace BatmanInfer {
    struct BINEQuantizationLayer::Impl {
        const BIITensor *src{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuQuantize> op{nullptr};
    };

    BINEQuantizationLayer::BINEQuantizationLayer() : _impl(std::make_unique<Impl>()) {
    }

    BINEQuantizationLayer::~BINEQuantizationLayer() = default;

    BIStatus BINEQuantizationLayer::validate(const BIITensorInfo *input, const BIITensorInfo *output) {
        return cpu::BICpuQuantize::validate(input, output);
    }

    void BINEQuantizationLayer::configure(const BIITensor *input, BIITensor *output) {
        _impl->src = input;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuQuantize>();
        _impl->op->configure(input->info(), output->info());
    }

    void BINEQuantizationLayer::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC, _impl->src);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }
} // namespace BatmanInfer