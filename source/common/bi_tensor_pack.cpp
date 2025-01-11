//
// Created by Mason on 2025/1/11.
//

#include <common/bi_tensor_pack.hpp>

#include <common/bi_i_tensor_v2.hpp>
#include <common/utils/bi_validate.hpp>

namespace BatmanInfer {
    BITensorPack::BITensorPack(BatmanInfer::BIIContext *ctx) : BclTensorPack_(), _pack() {
        BI_COMPUTE_ASSERT_NOT_NULLPTR(ctx);
        this->header.ctx = ctx;
        this->header.ctx->inc_ref();
    }

    BITensorPack::~BITensorPack() {
        this->header.ctx->dec_ref();
        this->header.type = detail::BIObjectType::Invalid;
    }

    BclStatus BITensorPack::add_tensor(BIITensorV2 *tensor, int32_t slot_id) {
        _pack.add_tensor(slot_id, tensor->tensor());
        return BclStatus::BclSuccess;
    }

    size_t BITensorPack::size() const {
        return _pack.size();
    }

    bool BITensorPack::empty() const {
        return _pack.empty();
    }

    bool BITensorPack::is_valid() const {
        return this->header.type == detail::BIObjectType::TensorPack;
    }

    BatmanInfer::BIITensor *BITensorPack::get_tensor(int32_t slot_id) {
        return _pack.get_tensor(slot_id);
    }

    BatmanInfer::BIITensorPack &BITensorPack::get_tensor_pack() {
        return _pack;
    }

}