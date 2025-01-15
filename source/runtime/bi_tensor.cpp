//
// Created by Mason on 2025/1/2.
//

#include <runtime/bi_tensor.hpp>

namespace BatmanInfer {
    BITensor::BITensor(BatmanInfer::BIIRuntimeContext *ctx) : _allocator(this) {

    }

    BIITensorInfo *BITensor::info() {
        return &_allocator.info();
    }

    BIITensorInfo *BITensor::info() const {
        return &_allocator.info();
    }

    uint8_t *BITensor::buffer() const {
        return _allocator.data();
    }

    BITensorAllocator *BITensor::allocator() {
        return &_allocator;
    }

    void BITensor::associate_memory_group(BIIMemoryGroup *memory_group) {
        _allocator.set_associated_memory_group(memory_group);
    }
}