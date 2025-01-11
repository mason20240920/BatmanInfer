//
// Created by Mason on 2025/1/11.
//

#include <bcl_entry_points.hpp>

#include <common/bi_i_tensor_v2.hpp>
#include <common/bi_tensor_pack.hpp>
#include <common/utils/bi_macros.hpp>

namespace {
    using namespace BatmanInfer;

    StatusCode PackTensorInternal(BITensorPack &pack, BclTensor external_tensor, int32_t slot_id) {
        auto status = StatusCode::Success;
        auto tensor = get_internal(external_tensor);

        status = detail::validate_internal_tensor(tensor);

        if (status != StatusCode::Success) {
            return status;
        }

        pack.add_tensor(tensor, slot_id);

        return status;
    }
} // namespace

extern "C" BclStatus BclCreateTensorPack(BclTensorPack *external_pack, BclContext external_ctx) {
    using namespace BatmanInfer;

    BIIContext *ctx = get_internal(external_ctx);

    const StatusCode status = detail::validate_internal_context(ctx);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    auto pack = new BITensorPack(ctx);
    if (pack == nullptr) {
        BI_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL("Couldn't allocate internal resources!");
        return BclOutOfMemory;
    }
    *external_pack = pack;

    return BclSuccess;
}

extern "C" BclStatus BclPackTensor(BclTensorPack external_pack, BclTensor external_tensor, int32_t slot_id) {
    using namespace BatmanInfer;

    auto pack = get_internal(external_pack);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(detail::validate_internal_pack(pack));
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(PackTensorInternal(*pack, external_tensor, slot_id));
    return BclStatus::BclSuccess;
}

extern "C" BclStatus
BclPackTensors(BclTensorPack external_pack, BclTensor *external_tensors, int32_t *slot_ids, size_t num_tensors) {
    using namespace BatmanInfer;

    auto pack = get_internal(external_pack);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(detail::validate_internal_pack(pack));

    for (unsigned i = 0; i < num_tensors; ++i) {
        BI_COMPUTE_RETURN_CENUM_ON_FAILURE(PackTensorInternal(*pack, external_tensors[i], slot_ids[i]));
    }
    return BclStatus::BclSuccess;
}

extern "C" BclStatus BclDestroyTensorPack(BclTensorPack external_pack) {
    using namespace BatmanInfer;

    auto pack = get_internal(external_pack);
    StatusCode status = detail::validate_internal_pack(pack);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    delete pack;

    return BclSuccess;
}