//
// Created by Mason on 2025/1/11.
//

#include <cpu/operators/bi_cpu_activation.hpp>

#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/bi_i_operator.hpp>
#include <common/utils/legacy_support.hpp>
#include <common/utils/bi_log.hpp>
#include <cpu/bi_cpu_context.hpp>
#include <cpu/kernels/bi_cpu_activation_kernel.hpp>

namespace BatmanInfer {
    namespace cpu {

        void BICpuActivation::configure(const BIITensorInfo *input,
                                        BIITensorInfo *output,
                                        const BIActivationLayerInfo &activation_info) {
            BI_COMPUTE_LOG_PARAMS(input, output);
            auto k = std::make_unique<kernels::BICpuActivationKernel>();
            k->configure(input, output, activation_info);
            _kernel = std::move(k);
        }

        BIStatus
        BICpuActivation::validate(const BIITensorInfo *input,
                                  const BIITensorInfo *output,
                                  const BIActivationLayerInfo &activation_info) {
            return kernels::BICpuActivationKernel::validate(input, output, activation_info);
        }

        void BICpuActivation::run(BIITensorPack &tensors) {
            BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
            auto split_dimension = static_cast<kernels::BICpuActivationKernel *>(_kernel.get())->get_split_dimension_hint();
            BINEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
        }


        std::tuple<BIIOperator *, StatusCode> BICpuContext::create_activation(const BclTensorDescriptor &src,
                                                                              const BclTensorDescriptor &dst,
                                                                              const BclActivationDescriptor &act,
                                                                              bool is_validate) {
            BITensorInfo src_info = detail::convert_to_legacy_tensor_info(src);
            BITensorInfo dst_info = detail::convert_to_legacy_tensor_info(dst);
            auto info = detail::convert_to_activation_info(act);

            if (is_validate &&
                !bool(BICpuActivation::validate(&src_info.set_is_resizable(false), &dst_info.set_is_resizable(false),
                                                info))) {
                return std::make_tuple(nullptr, StatusCode::UnsupportedConfig);
            }

            auto act_op = std::make_unique<cpu::BICpuActivation>();
            act_op->configure(&src_info, &dst_info, info);

            auto op = new BatmanInfer::BIIOperator(static_cast<BIIContext *>(this));
            if (op == nullptr) {
                BI_COMPUTE_LOG_ERROR_ACL("Couldn't allocate internal resources");
                return std::make_tuple(nullptr, StatusCode::OutOfMemory);
            }
            op->set_internal_operator(std::move(act_op));

            return std::make_tuple(op, StatusCode::Success);
        }
    }
}