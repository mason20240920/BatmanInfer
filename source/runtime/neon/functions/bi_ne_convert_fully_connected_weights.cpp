//
// Created by Mason on 2025/1/22.
//

#include <runtime/neon/functions/bi_ne_convert_fully_connected_weights.hpp>

#include <data/core/bi_vlidate.hpp>

#include <cpu/operators/bi_cpu_convert_fully_connected_weights.hpp>

namespace BatmanInfer {
    struct BINEConvertFullyConnectedWeights::Impl {
        const BIITensor *src{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuConvertFullyConnectedWeights> op{nullptr};
    };

    BINEConvertFullyConnectedWeights::BINEConvertFullyConnectedWeights() : _impl(std::make_unique<Impl>()) {

    }

    BINEConvertFullyConnectedWeights::~BINEConvertFullyConnectedWeights() = default;

    void
    BINEConvertFullyConnectedWeights::configure(const BatmanInfer::BIITensor *input,
                                                BatmanInfer::BIITensor *output,
                                                const BatmanInfer::BITensorShape &original_input_shape,
                                                BatmanInfer::BIDataLayout data_layout) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);

        _impl->src = input;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuConvertFullyConnectedWeights>();
        _impl->op->configure(_impl->src->info(), _impl->dst->info(), original_input_shape, data_layout);
    }

    void BINEConvertFullyConnectedWeights::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC, _impl->src);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }
}