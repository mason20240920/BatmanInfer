//
// Created by Mason on 2025/1/8.
//

#include <runtime/neon/functions/bi_ne_reshape_layer.hpp>

#include <data/core/bi_vlidate.hpp>

#include <cpu/operators/bi_cpu_reshape.hpp>

namespace BatmanInfer {
    struct BINEReshapeLayer::Impl {
        const BIITensor *src{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuReshape> op{nullptr};
    };

    BINEReshapeLayer::BINEReshapeLayer() : _impl(std::make_unique<Impl>()) {
    }

    BINEReshapeLayer::BINEReshapeLayer(BINEReshapeLayer &&) = default;

    BINEReshapeLayer &BINEReshapeLayer::operator=(BINEReshapeLayer &&) = default;

    BINEReshapeLayer::~BINEReshapeLayer() = default;

    void BINEReshapeLayer::configure(const BIITensor *input, BIITensor *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);

        _impl->src = input;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuReshape>();
        _impl->op->configure(input->info(), output->info());
    }

    void BINEReshapeLayer::dynamic_configure(BIITensor *output) {
        _impl->op->dynamic_configure(output->info());
    }


    BIStatus
    BINEReshapeLayer::validate(const BIITensorInfo *input, const BIITensorInfo *output) {
        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_RETURN_ON_ERROR(cpu::BICpuReshape::validate(input, output));

        return BIStatus{};
    }

    void BINEReshapeLayer::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC, _impl->src);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }
}
