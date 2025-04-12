//
// Created by Mason on 2025/2/7.
//

#include <cpu/operators/bi_cpu_quantize.hpp>

#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_cpu_quantize_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        BIStatus BICpuQuantize::validate(const BIITensorInfo *src, const BIITensorInfo *dst) {
            BI_COMPUTE_RETURN_ON_ERROR(kernels::BICpuQuantizeKernel::validate(src, dst));
            return BIStatus{};
        }

        void BICpuQuantize::configure(const BIITensorInfo *src, BIITensorInfo *dst) {
            BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
            BI_COMPUTE_LOG_PARAMS(src, dst);

            // Configure quantize kernel
            auto k = std::make_unique<kernels::BICpuQuantizeKernel>();
            k->configure(src, dst);
            _kernel = std::move(k);
        }

        void BICpuQuantize::dynamic_configure(const BIITensorInfo *src) const {
            auto k = reinterpret_cast<kernels::BICpuQuantizeKernel *>(_kernel.get());
            k->dynamic_configure(src);
        }


        void BICpuQuantize::run(BIITensorPack &tensors) {
            BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
            auto split_dimension = static_cast<kernels::BICpuQuantizeKernel *>(_kernel.get())->
                    get_split_dimension_hint();
            BINEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
        }
    } // namespace cpu
} // namespace BatmanInfer
