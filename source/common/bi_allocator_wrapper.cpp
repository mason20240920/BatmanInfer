//
// Created by Mason on 2025/1/10.
//

#include <common/bi_allocator_wrapper.hpp>

#include <data/core/bi_error.h>

namespace BatmanInfer {
    BIAllocatorWrapper::BIAllocatorWrapper(const BclAllocator &backing_allocator) noexcept:
            _backing_allocator(backing_allocator) {

    }

    void *BIAllocatorWrapper::alloc(size_t size) {
        BI_COMPUTE_ERROR_ON(_backing_allocator.alloc == nullptr);
        return _backing_allocator.alloc(_backing_allocator.user_data, size);
    }

    void BIAllocatorWrapper::free(void *ptr) {
        BI_COMPUTE_ERROR_ON(_backing_allocator.free == nullptr);
        _backing_allocator.free(_backing_allocator.user_data, ptr);
    }

    void *BIAllocatorWrapper::aligned_alloc(size_t size, size_t alignment) {
        BI_COMPUTE_ERROR_ON(_backing_allocator.aligned_alloc == nullptr);
        return _backing_allocator.aligned_alloc(_backing_allocator.user_data, size, alignment);
    }

    void BIAllocatorWrapper::aligned_free(void *ptr) {
        BI_COMPUTE_ERROR_ON(_backing_allocator.aligned_free == nullptr);
        _backing_allocator.aligned_free(_backing_allocator.user_data, ptr);
    }

    void BIAllocatorWrapper::set_user_data(void *user_data) {
        if (user_data != nullptr)
            _backing_allocator.user_data = user_data;
    }
}