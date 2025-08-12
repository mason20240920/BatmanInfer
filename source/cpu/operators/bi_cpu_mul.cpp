//
// Created by Mason on 2025/1/17.
//

#include <cpu/operators/bi_cpu_mul.hpp>

#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_cpu_mul_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {
        BIStatus BICpuMul::validate(const BatmanInfer::BIITensorInfo *src1, const BatmanInfer::BIITensorInfo *src2,
                                    const BatmanInfer::BIITensorInfo *dst, float scale,
                                    BatmanInfer::BIConvertPolicy overflow_policy,
                                    BatmanInfer::BIRoundingPolicy rounding_policy,
                                    const BatmanInfer::BIActivationLayerInfo &act_info) {
            BI_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
            return kernels::BICpuMulKernel::validate(src1, src2, dst, scale, overflow_policy, rounding_policy);
        }

        void BICpuMul::configure(BatmanInfer::BIITensorInfo *src1, BatmanInfer::BIITensorInfo *src2,
                                 BatmanInfer::BIITensorInfo *dst, float scale,
                                 BatmanInfer::BIConvertPolicy overflow_policy,
                                 BatmanInfer::BIRoundingPolicy rounding_policy,
                                 const BatmanInfer::BIActivationLayerInfo &act_info) {
            BI_COMPUTE_UNUSED(act_info);
            BI_COMPUTE_LOG_PARAMS(src1, src2, dst, scale, overflow_policy, rounding_policy, act_info);

            auto k = std::make_unique<kernels::BICpuMulKernel>();
            k->configure(src1, src2, dst, scale, overflow_policy, rounding_policy);
            _kernel = std::move(k);
        }

        void BICpuMul::dynamic_configure(BIITensorInfo *src1, BIITensorInfo *src2) const {
            auto k = reinterpret_cast<kernels::BICpuMulKernel *>(_kernel.get());
            k->dynamic_configure(src1, src2);
        }


        void BICpuMul::run(BatmanInfer::BIITensorPack &tensors) {
            BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
            auto split_dimension = static_cast<kernels::BICpuMulKernel *>(_kernel.get())->get_split_dimension_hint();
            BINEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
        }

        BIStatus BICpuComplexMul::validate(const BIITensorInfo *src1,
                                           const BIITensorInfo *src2,
                                           const BIITensorInfo *dst,
                                           const BIActivationLayerInfo &act_info) {
            BI_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
            return kernels::BICpuComplexMulKernel::validate(src1, src2, dst);
        }

        void BICpuComplexMul::configure(BIITensorInfo *src1,
                                        BIITensorInfo *src2,
                                        BIITensorInfo *dst,
                                        const BIActivationLayerInfo &act_info) {
            BI_COMPUTE_UNUSED(act_info);
            BI_COMPUTE_LOG_PARAMS(src1, src2, dst, act_info);

            auto k = std::make_unique<kernels::BICpuComplexMulKernel>();
            k->configure(src1, src2, dst);
            _kernel = std::move(k);
        }

        void BICpuComplexMul::run(BIITensorPack &tensors) {
            BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
            BINEScheduler::get().schedule_op(_kernel.get(), BIWindow::DimY, _kernel->window(), tensors);
        }
    }
}