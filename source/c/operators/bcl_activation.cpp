//
// Created by Mason on 2025/1/11.
//

#ifndef BATMANINFER_BCL_ACTIVATION_CPP
#define BATMANINFER_BCL_ACTIVATION_CPP

#include <bcl_operators.hpp>

#include <common/bi_i_operator.hpp>
#include <common/utils/bi_macros.hpp>
#include <common/utils/bi_validate.hpp>

extern "C" BclStatus AclActivation(BclOperator *external_op,
                                   BclContext external_ctx,
                                   const BclTensorDescriptor *src,
                                   const BclTensorDescriptor *dst,
                                   const BclActivationDescriptor info) {
    using namespace BatmanInfer;

    // Extract internal context
    auto ctx = get_internal(external_ctx);
    StatusCode status = detail::validate_internal_context(ctx);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    bool is_validate = (external_op == BI_COMPUTE_VALIDATE_OPERATOR_SUPPORT);

    std::tie(*external_op, status) = ctx->create_activation(*src, *dst, info, is_validate);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    return BclSuccess;
}

#endif //BATMANINFER_BCL_ACTIVATION_CPP
