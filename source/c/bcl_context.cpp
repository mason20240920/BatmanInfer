//
// Created by Mason on 2025/1/11.
//

#include <bcl_entry_points.hpp>
#include <data/core/bi_error.h>

#include <common/bi_i_context.hpp>
#include <common/utils/bi_macros.hpp>
#include <common/utils/bi_validate.hpp>

#ifdef BI_COMPUTE_CPU_ENABLED

#include  <cpu/bi_cpu_context.hpp>

#endif /* ARM_COMPUTE_CPU_ENABLED */

#ifdef BI_COMPUTE_OPENCL_ENABLED
#include "src/gpu/cl/ClContext.h"
#endif /* ARM_COMPUTE_OPENCL_ENABLED */

namespace {
    template<typename ContextType>
    BatmanInfer::BIIContext *create_backend_ctx(const BclContextOptions *options) {
        return new(std::nothrow) ContextType(options);
    }

    bool is_target_valid(BclTarget target) {
        return BatmanInfer::utils::is_in(target, {BclCpu, BclGpuOcl});
    }

    bool are_context_options_valid(const BclContextOptions *options) {
        BI_COMPUTE_ASSERT_NOT_NULLPTR(options);
        return BatmanInfer::utils::is_in(options->mode, {BclPreferFastRerun, BclPreferFastStart});
    }

    BatmanInfer::BIIContext *create_context(BclTarget target, const BclContextOptions *options) {
        BI_COMPUTE_UNUSED(options);

        switch (target) {
#ifdef BI_COMPUTE_CPU_ENABLED
            case BclCpu:
                return create_backend_ctx<BatmanInfer::cpu::BICpuContext>(options);
#endif /* ARM_COMPUTE_CPU_ENABLED */
#ifdef BI_COMPUTE_OPENCL_ENABLED
                case BclGpuOcl:
                return create_backend_ctx<arm_compute::gpu::opencl::ClContext>(options);
#endif /* ARM_COMPUTE_OPENCL_ENABLED */
            default:
                return nullptr;
        }
        return nullptr;
    }
} // namespace

extern "C" BclStatus BclCreateContext(BclContext *external_ctx, BclTarget target, const BclContextOptions *options) {
    if (!is_target_valid(target)) {
        BI_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL("Target is invalid!");
        return BclUnsupportedTarget;
    }

    if (options != nullptr && !are_context_options_valid(options)) {
        BI_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL("Context options are invalid!");
        return BclInvalidArgument;
    }

    auto ctx = create_context(target, options);
    if (ctx == nullptr) {
        BI_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL("Couldn't allocate internal resources for context creation!");
        return BclOutOfMemory;
    }
    *external_ctx = ctx;

    return BclSuccess;
}

extern "C" BclStatus BclDestroyContext(BclContext external_ctx) {
    using namespace BatmanInfer;

    BIIContext *ctx = get_internal(external_ctx);

    StatusCode status = detail::validate_internal_context(ctx);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    if (ctx->refcount() != 0) {
        BI_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL("Context has references on it that haven't been released!");
        // TODO: Fix the refcount with callback when reaches 0
    }

    delete ctx;

    return utils::as_cenum<BclStatus>(status);
}
