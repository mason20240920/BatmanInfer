//
// Created by Mason on 2025/1/10.
//

#include <common/bi_i_operator.hpp>

#include <common/utils/bi_validate.hpp>

namespace BatmanInfer {
#ifndef DOXYGEN_SKIP_THIS

    BIIOperator::BIIOperator(BatmanInfer::BIIContext *ctx) {
        BI_COMPUTE_ASSERT_NOT_NULLPTR(ctx);
        this->header.ctx = ctx;
        this->header.ctx->inc_ref();
    }

    BIIOperator::~BIIOperator() {
        this->header.ctx->dec_ref();
        this->header.type = detail::BIObjectType::Invalid;
    }

    bool BIIOperator::is_valid() const {
        return this->header.type == detail::BIObjectType::Operator;
    }

    StatusCode BIIOperator::run(BIITensorPack &tensors) {
        _op->run(tensors);
        return StatusCode::Success;
    }

    StatusCode BIIOperator::run(BIIQueue &queue,
                                BIITensorPack &tensors) {
        BI_COMPUTE_UNUSED(queue);
        _op->run(tensors);
        return StatusCode::Success;
    }

    StatusCode BIIOperator::prepare(BatmanInfer::BIITensorPack &tensors) {
        _op->prepare(tensors);
        return StatusCode::Success;
    }

    MemoryRequirements BIIOperator::workspace() const {
        return _op->workspace();
    }

#endif  /* DOXYGEN_SKIP_THIS */
}