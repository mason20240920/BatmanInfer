//
// Created by Mason on 2025/2/8.
//

#include <cpu/operators/bi_cpu_dequantize.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_cpu_dequantize_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        void BICpuDequantize::configure(const BatmanInfer::BIITensorInfo *src, BatmanInfer::BIITensorInfo *dst) {
            BI_COMPUTE_LOG_PARAMS(src, dst);
            auto k = std::make_unique<kernels::BICpuDequantizeKernel>();
            k->configure(src, dst);
            _kernel = std::move(k);
        }

        BIStatus
        BICpuDequantize::validate(const BatmanInfer::BIITensorInfo *src, const BatmanInfer::BIITensorInfo *dst) {
            return kernels::BICpuDequantizeKernel::validate(src, dst);
        }

        void BICpuDequantize::dynamic_configure(const BIITensorInfo *src) const {
            const auto k = reinterpret_cast<kernels::BICpuDequantizeKernel *>(_kernel.get());
            k->dynamic_configure(src);
        }


        void BICpuDequantize::run(BatmanInfer::BIITensorPack &tensors) {
            BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
            prepare(tensors);
            BINEScheduler::get().schedule_op(_kernel.get(), BIWindow::DimY, _kernel->window(), tensors);
        }
    } // namespace cpu
}
