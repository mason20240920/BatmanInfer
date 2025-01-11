//
// Created by Mason on 2025/1/11.
//

#include <bcl_entry_points.hpp>

#include <common/bi_i_queue.hpp>
#include <common/utils/bi_macros.hpp>
#include <common/utils/bi_validate.hpp>

namespace {
    /** Check if queue options are valid
 *
 * @param[in] options Queue options
 *
 * @return true in case of success else false
 */
    bool is_mode_valid(const BclQueueOptions *options) {
        BI_COMPUTE_ASSERT_NOT_NULLPTR(options);
        return BatmanInfer::utils::is_in(options->mode, {BclTuningModeNone, BclRapid, BclNormal, BclExhaustive});
    }
} // namespace

extern "C" BclStatus BclCreateQueue(BclQueue *external_queue, BclContext external_ctx, const BclQueueOptions *options) {
    using namespace BatmanInfer;

    auto ctx = get_internal(external_ctx);

    StatusCode status = detail::validate_internal_context(ctx);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    if (options != nullptr && !is_mode_valid(options)) {
        BI_COMPUTE_LOG_ERROR_ACL("Queue options are invalid");
        return BclInvalidArgument;
    }

    auto queue = ctx->create_queue(options);
    if (queue == nullptr) {
        BI_COMPUTE_LOG_ERROR_ACL("Couldn't allocate internal resources");
        return BclOutOfMemory;
    }

    *external_queue = queue;

    return BclSuccess;
}

extern "C" BclStatus BclQueueFinish(BclQueue external_queue) {
    using namespace BatmanInfer;

    auto queue = get_internal(external_queue);

    StatusCode status = detail::validate_internal_queue(queue);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    status = queue->finish();
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    return BclSuccess;
}

extern "C" BclStatus BclDestroyQueue(BclQueue external_queue) {
    using namespace BatmanInfer;

    auto queue = get_internal(external_queue);

    StatusCode status = detail::validate_internal_queue(queue);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    delete queue;

    return BclSuccess;
}