//
// Created by Mason on 2025/1/17.
//

#include <runtime/neon/functions/ne_pixel_wise_multiplication.hpp>

#include <data/core/bi_i_tensor.hpp>

#include <cpu/operators/bi_cpu_mul.hpp>

namespace BatmanInfer {
    struct BINEPixelWiseMultiplication::Impl {
        const BIITensor *src_0{nullptr};
        const BIITensor *src_1{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuMul> op{nullptr};
    };

    BINEPixelWiseMultiplication::BINEPixelWiseMultiplication() : _impl(std::make_unique<Impl>()) {

    }

    BINEPixelWiseMultiplication::~BINEPixelWiseMultiplication() = default;

    BIStatus BINEPixelWiseMultiplication::validate(const BatmanInfer::BIITensorInfo *input1,
                                                   const BatmanInfer::BIITensorInfo *input2,
                                                   const BatmanInfer::BIITensorInfo *output, float scale,
                                                   BatmanInfer::BIConvertPolicy overflow_policy,
                                                   BatmanInfer::BIRoundingPolicy rounding_policy,
                                                   const BatmanInfer::BIActivationLayerInfo &act_info) {
        return cpu::BICpuMul::validate(input1, input2, output, scale, overflow_policy, rounding_policy, act_info);
    }

    void BINEPixelWiseMultiplication::configure(const BIITensor *input1,
                                                const BIITensor *input2,
                                                BIITensor *output,
                                                float scale,
                                                BIConvertPolicy overflow_policy,
                                                BIRoundingPolicy rounding_policy,
                                                const BIActivationLayerInfo &act_info) {
        _impl->src_0 = input1;
        _impl->src_1 = input2;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuMul>();
        _impl->op->configure(input1->info(), input2->info(), output->info(), scale, overflow_policy, rounding_policy,
                             act_info);
    }

    void BINEPixelWiseMultiplication::dynamic_configure() const {
        _impl->op->dynamic_configure(_impl->src_0->info(),  _impl->src_1->info());
    }


    void BINEPixelWiseMultiplication::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC_0, _impl->src_0);
        pack.add_tensor(BITensorType::ACL_SRC_1, _impl->src_1);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }

    struct BINEComplexPixelWiseMultiplication::Impl {
        BIITensor *src_0{nullptr};
        BIITensor *src_1{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuComplexMul> op{nullptr};
    };

    BINEComplexPixelWiseMultiplication::BINEComplexPixelWiseMultiplication() : _impl(std::make_unique<Impl>()) {
    }

    BINEComplexPixelWiseMultiplication::~BINEComplexPixelWiseMultiplication() = default;

    BIStatus BINEComplexPixelWiseMultiplication::validate(const BIITensorInfo *input1,
                                                          const BIITensorInfo *input2,
                                                          const BIITensorInfo *output,
                                                          const BIActivationLayerInfo &act_info) {
        return cpu::BICpuComplexMul::validate(input1, input2, output, act_info);
    }

    void BINEComplexPixelWiseMultiplication::configure(BIITensor *input1,
                                                       BIITensor *input2,
                                                       BIITensor *output,
                                                       const BIActivationLayerInfo &act_info) {
        _impl->src_0 = input1;
        _impl->src_1 = input2;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuComplexMul>();
        _impl->op->configure(input1->info(), input2->info(), output->info(), act_info);
    }

    void BINEComplexPixelWiseMultiplication::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC_0, _impl->src_0);
        pack.add_tensor(BITensorType::ACL_SRC_1, _impl->src_1);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }
}