//
// Created by Mason on 2024/12/30.
//

#include <runtime/bi_blob_lifetime_manager.hpp>

#include <data/core/bi_error.h>

#include <runtime/bi_blob_memory_pool.hpp>

namespace BatmanInfer {
    BIBlobLifetimeManager::BIBlobLifetimeManager() : _blobs() {

    }

    const BIBlobLifetimeManager::info_type &BIBlobLifetimeManager::info() const {
        return _blobs;
    }

    std::unique_ptr<BIIMemoryPool> BIBlobLifetimeManager::create_pool(BatmanInfer::BIIAllocator *allocator) {
        BI_COMPUTE_ERROR_ON(allocator == nullptr);
        return std::make_unique<BIBlobMemoryPool>(allocator, _blobs);
    }

    BIMappingType BIBlobLifetimeManager::mapping_type() const {
        return BIMappingType::BLOBS;
    }

    void BIBlobLifetimeManager::update_blobs_and_mappings() {
        BI_COMPUTE_ERROR_ON(!are_all_finalized());
        BI_COMPUTE_ERROR_ON(_active_group == nullptr);

        // 按照降序排列自由 blob 的要求。
        _free_blobs.sort([](const Blob &ba, const Blob &bb) { return ba.max_size > bb.max_size;});

        // 创建 内存组的大小数组
        std::vector<BIBlobInfo> group_sizes;
        std::transform(std::begin(_free_blobs), std::end(_free_blobs), std::back_inserter(group_sizes),
                       [](const Blob &b) {
            return BIBlobInfo{b.max_size, b.max_alignment, b.bound_elements.size()};
        });
    }
}