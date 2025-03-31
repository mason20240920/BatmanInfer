//
// Created by Mason on 2025/2/12.
//

#include <runtime/neon/functions/BINERMSNormLayer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/BINERMSNormLayerKernel.hpp>

namespace BatmanInfer {
    BINERMSNormLayer::~BINERMSNormLayer() = default;

    BINERMSNormLayer::BINERMSNormLayer(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(
            std::move(memory_manager)), _rms_norm_kernel() {}

    void BINERMSNormLayer::configure(const BatmanInfer::BIITensor *input, const BatmanInfer::BIITensor *gamma,
                                     BatmanInfer::BIITensor *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, gamma, output);
        BI_COMPUTE_LOG_PARAMS(input, gamma, output);

        _rms_norm_kernel = std::make_unique<cpu::BINERMSNormLayerKernel>();
        _rms_norm_kernel->configure(input, gamma, output);

    }

    BIStatus
    BINERMSNormLayer::validate(const BatmanInfer::BIITensorInfo *input, const BatmanInfer::BIITensorInfo *gamma,
                               const BatmanInfer::BIITensorInfo *output) {
        // Perform validation step
        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, gamma);

        BI_COMPUTE_RETURN_ON_ERROR(cpu::BINERMSNormLayerKernel::validate(input, gamma, output));

        return BIStatus{};
    }

    void BINERMSNormLayer::run() {
        BIMemoryGroupResourceScope scope_mg(_memory_group);
        _rms_norm_kernel->run(_rms_norm_kernel->window(), ThreadInfo{});
    }
}