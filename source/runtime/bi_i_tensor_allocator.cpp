//
// Created by Mason on 2024/12/31.
//

#include <runtime/bi_i_tensor_allocator.hpp>

using namespace BatmanInfer;

void BIITensorAllocator::init(const BatmanInfer::BITensorInfo &input, size_t alignment) {
    _info_owned = input;
    _info_external = nullptr;
    _alignment = alignment;
}

void BIITensorAllocator::soft_init(BatmanInfer::BITensorInfo &input, size_t alignment) {
    _info_external = &input;
    _alignment = alignment;
}

BITensorInfo &BIITensorAllocator::info() {
    return (_info_external != nullptr) ? *_info_external : _info_owned;
}

const BITensorInfo &BIITensorAllocator::info() const {
    return (_info_external != nullptr) ? *_info_external : _info_owned;
}

size_t BIITensorAllocator::alignment() const {
    return _alignment;
}