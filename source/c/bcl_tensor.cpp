//
// Created by Mason on 2025/1/10.
//

#include <bcl_entry_points.hpp>
#include <bcl_utils.hpp>
#include <data/core/bi_error.h>

#include <common/bi_i_tensor_v2.hpp>
#include <common/utils/bi_macros.hpp>

namespace {
    using namespace BatmanInfer;

    /**
     * 计算库最大的维度
     */
    constexpr int32_t max_allowed_dims = 6;

    /**
     * 检查一个描述是否合理
     *
     * @param desc 需要验证的描述器
     * @return
     */
    bool is_desc_valid(const BclTensorDescriptor &desc) {
        if (desc.data_type > BclFloat32 || desc.data_type <= BclDataTypeUnknown) {
            BI_COMPUTE_LOG_ERROR_ACL("[BclCreateTensor]: Unknown data type!");
            return false;
        }
        if (desc.ndims > max_allowed_dims) {
            BI_COMPUTE_LOG_ERROR_ACL("[BclCreateTensor]: Dimensions surpass the maximum allowed value!");
            return false;
        }
        if (desc.ndims > 0 && desc.shape == nullptr) {
            BI_COMPUTE_LOG_ERROR_ACL("[BclCreateTensor]: Dimensions values are empty while dimensionality is > 0!");
            return false;
        }
        return true;
    }

    StatusCode convert_and_validate_tensor(BclTensor tensor, BIITensorV2 **internal_tensor) {
        *internal_tensor = get_internal(tensor);
        return detail::validate_internal_tensor(*internal_tensor);
    }

}

extern "C" BclStatus
BclCreateTensor(BclTensor *external_tensor, BclContext external_ctx, const BclTensorDescriptor *desc, bool allocate) {
    using namespace BatmanInfer;

    BIIContext *ctx = get_internal(external_ctx);

    StatusCode status = detail::validate_internal_context(ctx);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    if (desc == nullptr || !is_desc_valid(*desc)) {
        BI_COMPUTE_LOG_ERROR_ACL("[BclCreateTensor]: Descriptor is invalid!");
        return BclInvalidArgument;
    }

    auto tensor = ctx->create_tensor(*desc, allocate);
    if (tensor == nullptr) {
        BI_COMPUTE_LOG_ERROR_ACL("[BclCreateTensor]: Couldn't allocate internal resources for tensor creation!");
        return BclOutOfMemory;
    }
    *external_tensor = tensor;

    return BclSuccess;
}

extern "C" BclStatus BclMapTensor(BclTensor external_tensor, void **handle) {
    using namespace BatmanInfer;

    auto tensor = get_internal(external_tensor);
    StatusCode status = detail::validate_internal_tensor(tensor);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    if (handle == nullptr) {
        BI_COMPUTE_LOG_ERROR_ACL("[BclMapTensor]: Handle object is nullptr!");
        return BclInvalidArgument;
    }

    *handle = tensor->map();

    return BclSuccess;
}

extern "C" BclStatus BclUnmapTensor(BclTensor external_tensor, void *handle) {
    BI_COMPUTE_UNUSED(handle);

    using namespace BatmanInfer;

    auto tensor = get_internal(external_tensor);
    StatusCode status = detail::validate_internal_tensor(tensor);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    status = tensor->unmap();
    return BclSuccess;
}

extern "C" BclStatus BclTensorImport(BclTensor external_tensor,
                                     void *handle,
                                     BclImportMemoryType type) {
    using namespace BatmanInfer;

    auto tensor = get_internal(external_tensor);
    StatusCode status = detail::validate_internal_tensor(tensor);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    status = tensor->import(handle, utils::as_enum<ImportMemoryType>(type));
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    return BclSuccess;
}

extern "C" BclStatus BclDestroyTensor(BclTensor external_tensor) {
    using namespace BatmanInfer;

    auto tensor = get_internal(external_tensor);

    StatusCode status = detail::validate_internal_tensor(tensor);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    delete tensor;

    return BclSuccess;
}

extern "C" BclStatus BclGetTensorSize(BclTensor tensor,
                                      uint64_t *size) {
    using namespace BatmanInfer;

    if (size == nullptr)
        return BclStatus::BclInvalidArgument;

    BIITensorV2 *internal_tensor{nullptr};
    auto status = convert_and_validate_tensor(tensor, &internal_tensor);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    *size = internal_tensor->get_size();
    return utils::as_cenum<BclStatus>(status);
}

extern "C" BclStatus BclGetTensorDescriptor(BclTensor tensor,
                                            BclTensorDescriptor *desc) {
    using namespace BatmanInfer;

    if (desc == nullptr)
        return BclStatus::BclInvalidArgument;

    BIITensorV2 *internal_tensor{nullptr};
    const auto status = convert_and_validate_tensor(tensor, &internal_tensor);
    BI_COMPUTE_RETURN_CENUM_ON_FAILURE(status);

    *desc = internal_tensor->get_descriptor();
    return utils::as_cenum<BclStatus>(status);
}
