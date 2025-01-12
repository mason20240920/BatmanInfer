//
// Created by Mason on 2025/1/12.
//

#include <cpu/operators/bi_cpu_add.hpp>

#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_cpu_add_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        void BICpuAdd::configure(const BIITensorInfo *src0,
                                 const BIITensorInfo *src1,
                                 BIITensorInfo *dst,
                                 BIConvertPolicy policy,
                                 const BIActivationLayerInfo &act_info) {
            BI_COMPUTE_UNUSED(act_info);
//            BI_COMPUTE_LOG_PARAMS(src0, src1, dst, policy, act_info);
            auto k = std::make_unique<kernels::BICpuAddKernel>();
            k->configure(src0, src1, dst, policy);
            _kernel = std::move(k);
        }

        BIStatus BICpuAdd::validate(const BatmanInfer::BIITensorInfo *src0, const BatmanInfer::BIITensorInfo *src1,
                                    const BatmanInfer::BIITensorInfo *dst, BatmanInfer::BIConvertPolicy policy,
                                    const BatmanInfer::BIActivationLayerInfo &act_info) {
            BI_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
            return kernels::BICpuAddKernel::validate(src0, src1, dst, policy);
        }

        void BICpuAdd::run(BatmanInfer::BIITensorPack &tensors) {
            const auto split_dimension = static_cast<kernels::BICpuAddKernel *>(_kernel.get())->get_split_dimension();

            BINEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
        }
    }
}