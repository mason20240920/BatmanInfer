//
// Created by Mason on 2024/12/26.
//

#include <runtime/bi_allocator.hpp>
#include "arm_compute/core/Error.h"
#include <runtime/bi_memory_region.hpp>

using namespace BatmanInfer;

void *BIAllocator::allocate(size_t size, size_t alignment) {
    ARM_COMPUTE_UNUSED(alignment);
    return ::operator new(size);
}

void BIAllocator::free(void *ptr) {
    ::operator delete(ptr);
}

std::unique_ptr<BIIMemoryRegion> BIAllocator::make_region(size_t size, size_t alignment) {
    return std::make_unique<BIMemoryRegion>(size, alignment);
}