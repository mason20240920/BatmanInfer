//
// Created by Mason on 2025/1/11.
//

#include "bcl_entry_points.hpp"

#include <common/bi_i_operator.hpp>
#include <common/bi_i_queue.hpp>
#include <common/bi_tensor_pack.hpp>
#include <common/utils/bi_macros.hpp>

extern "C" BclStatus BclRunOperator(BclOperator external_op, BclQueue external_queue, BclTensorPack external_tensors) {
    using namespace BatmanInfer;

    auto op = get_internal(external_op);
    auto queue = get_internal(external_queue);
    auto pack = get_internal(external_tensors);

    StatusCode status = StatusCode::Success;
    status = detail::validate_internal_operator(op);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);
    status = detail::validate_internal_queue(queue);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);
    status = detail::validate_internal_pack(pack);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    status = op->run(*queue, pack->get_tensor_pack());
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    return BclSuccess;
}

extern "C" BclStatus BclDestroyOperator(BclOperator external_op) {
    using namespace BatmanInfer;

    auto op = get_internal(external_op);

    StatusCode status = detail::validate_internal_operator(op);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    delete op;

    return BclSuccess;
}
