//
// Created by Mason on 2024/12/30.
//

#include <runtime/bi_blob_memory_pool.hpp>
#include <utility>
#include <runtime/bi_i_allocator.hpp>

#include <data/core/bi_error.h>

using namespace BatmanInfer;

BIBlobMemoryPool::BIBlobMemoryPool(BatmanInfer::BIIAllocator *allocator, std::vector<BIBlobInfo> blob_info) :
        _allocator(allocator),
        _blobs(),
        _blob_info(std::move(blob_info)) {

}

BIBlobMemoryPool::~BIBlobMemoryPool() {
    free_blobs();
}


void BIBlobMemoryPool::acquire(BatmanInfer::BIMemoryMappings &handles) {
    // 设置内存处理器
    for (auto &handle: handles) {
        BI_COMPUTE_ERROR_ON(handle.first == nullptr);
        handle.first->set_region(_blobs[handle.second].get());
    }
}

void BIBlobMemoryPool::release(BatmanInfer::BIMemoryMappings &handles) {
    for (auto &handle: handles) {
        BI_COMPUTE_ERROR_ON(handle.first == nullptr);
        handle.first->set_region(nullptr);
    }
}

BIMappingType BIBlobMemoryPool::mapping_type() const {
    return BIMappingType::BLOBS;
}

std::unique_ptr<BIIMemoryPool> BIBlobMemoryPool::duplicate() {
    BI_COMPUTE_ERROR_ON(!_allocator);
    return std::make_unique<BIBlobMemoryPool>(_allocator, _blob_info);
}

void BIBlobMemoryPool::allocate_blobs(const std::vector<BIBlobInfo> &blob_info) {
    BI_COMPUTE_ERROR_ON(!_allocator);

    for (const auto &bi: blob_info)
        _blobs.push_back(_allocator->make_region(bi.size, bi.alignment));
}

void BIBlobMemoryPool::free_blobs() {
    _blobs.clear();
}
