//
// Created by Mason on 2025/8/2.
//
//
// Created by Mason on 2025/2/12.
//

#include <runtime/neon/functions/BINELayerNormLayer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/BINELayerNormLayerKernel.hpp>

namespace BatmanInfer {
    BINELayerNormLayer::~BINELayerNormLayer() = default;

    BINELayerNormLayer::BINELayerNormLayer() : _layer_norm_kernel() {
    }

    void BINELayerNormLayer::configure(const BIITensor *input,
                                       const BIITensor *gamma,
                                       const BIITensor *beta,
                                       BIITensor *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, gamma, output, beta);
        BI_COMPUTE_LOG_PARAMS(input, gamma, output, beta);

        _layer_norm_kernel = std::make_unique<cpu::BINELayerNormLayerKernel>();
        _layer_norm_kernel->configure(input, gamma, beta, output);
    }


    BIStatus BINELayerNormLayer::validate(const BIITensorInfo *input, const BIITensorInfo *gamma, const BIITensor *beta,
                                          const BIITensorInfo *output) {
        // Perform validation step
        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, gamma, beta);

        BI_COMPUTE_RETURN_ON_ERROR(cpu::BINELayerNormLayerKernel::validate(input, gamma, output));

        return BIStatus{};
    }

    void BINELayerNormLayer::dynamic_configure(const BIITensor *input) const {
        _layer_norm_kernel->dynamic_configure(input);
    }


    void BINELayerNormLayer::run() {
        // _rms_norm_kernel->run(_rms_norm_kernel->window(), ThreadInfo{});
        BINEScheduler::get().schedule(_layer_norm_kernel.get(), BIWindow::DimY);
    }
}