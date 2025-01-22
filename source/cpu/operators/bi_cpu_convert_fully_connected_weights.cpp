//
// Created by Mason on 2025/1/22.
//

#include <cpu/operators/bi_cpu_convert_fully_connected_weights.hpp>

#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_cpu_convert_fully_connected_weights_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        void BICpuConvertFullyConnectedWeights::configure(const BatmanInfer::BIITensorInfo *src,
                                                          BatmanInfer::BIITensorInfo *dst,
                                                          const BatmanInfer::BITensorShape &original_src_shape,
                                                          BIDataLayout data_layout) {
            BI_COMPUTE_LOG_PARAMS(src, dst, original_src_shape, data_layout);
            auto k = std::make_unique<kernels::BICpuConvertFullyConnectedWeightsKernel>();
            k->configure(src, dst, original_src_shape, data_layout);
            _kernel = std::move(k);
        }

        BIStatus BICpuConvertFullyConnectedWeights::validate(const BatmanInfer::BIITensorInfo *src,
                                                             const BatmanInfer::BIITensorInfo *dst,
                                                             const BatmanInfer::BITensorShape &original_src_shape,
                                                             BatmanInfer::BIDataLayout data_layout) {
            return kernels::BICpuConvertFullyConnectedWeightsKernel::validate(src, dst, original_src_shape,
                                                                              data_layout);
        }

        void BICpuConvertFullyConnectedWeights::run(BatmanInfer::BIITensorPack &tensors) {
            BINEScheduler::get().schedule_op(_kernel.get(), BIWindow::DimZ, _kernel->window(), tensors);
        }
    }
}