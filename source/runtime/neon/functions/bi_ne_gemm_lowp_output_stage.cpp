//
// Created by Mason on 2025/2/7.
//

#include <runtime/neon/functions/bi_ne_gemm_lowp_output_stage.hpp>

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_vlidate.hpp>

#include <cpu/operators/bi_cpu_gemm_lowp_output_stage.hpp>

namespace BatmanInfer {
    struct BINEGEMMLowpOutputStage::Impl {
        const BIITensor *src{nullptr};
        const BIITensor *bias{nullptr};
        BIITensor *dst{nullptr};
        BIITensorPack run_pack{};
        std::unique_ptr<cpu::BICpuGemmLowpOutputStage> op{nullptr};
    };

    BINEGEMMLowpOutputStage::BINEGEMMLowpOutputStage() : _impl(std::make_unique<Impl>()) {
    }

    BINEGEMMLowpOutputStage::~BINEGEMMLowpOutputStage() = default;

    void BINEGEMMLowpOutputStage::configure(const BIITensor *input,
                                            const BIITensor *bias,
                                            BIITensor *output,
                                            const BIGEMMLowpOutputStageInfo &info) {
        // Perform validate step
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_ERROR_THROW_ON(
                BINEGEMMLowpOutputStage::validate(input->info(), bias != nullptr ? bias->info() : nullptr,
                                                  output->info(),
                                                  info));
        _impl->src = input;
        _impl->bias = bias;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuGemmLowpOutputStage>();
        _impl->op->configure(input->info(), (bias == nullptr) ? nullptr : bias->info(), output->info(), info);

        _impl->run_pack = {
                {BITensorType::ACL_SRC,  _impl->src},
                {BITensorType::ACL_BIAS, _impl->bias},
                {BITensorType::ACL_DST,  _impl->dst}};
    }

    BIStatus BINEGEMMLowpOutputStage::validate(const BIITensorInfo *input,
                                               const BIITensorInfo *bias,
                                               const BIITensorInfo *output,
                                               const BIGEMMLowpOutputStageInfo &info) {
        return cpu::BICpuGemmLowpOutputStage::validate(input, bias, output, info);
    }

    void BINEGEMMLowpOutputStage::run() {
        _impl->op->run(_impl->run_pack);
    }
}