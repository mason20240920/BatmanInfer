//
// Created by Mason on 2024/12/30.
//

#include <runtime/bi_blob_lifetime_manager.hpp>

#include <data/core/bi_error.h>

#include <runtime/bi_blob_memory_pool.hpp>

#include <runtime/bi_i_memory_group.hpp>

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
        _free_blobs.sort([](const Blob &ba, const Blob &bb) { return ba.max_size > bb.max_size; });

        // 创建 内存组的大小数组
        std::vector<BIBlobInfo> group_sizes;
        std::transform(std::begin(_free_blobs), std::end(_free_blobs), std::back_inserter(group_sizes),
                       [](const Blob &b) {
                           return BIBlobInfo{b.max_size, b.max_alignment, b.bound_elements.size()};
                       });

        // Update blob sizes
        size_t max_size = std::max(_blobs.size(), group_sizes.size());
        _blobs.resize(max_size);
        group_sizes.resize(max_size);
        std::transform(std::begin(_blobs), std::end(_blobs), std::begin(group_sizes), std::begin(_blobs),
                       [](BIBlobInfo lhs, BIBlobInfo rhs) {
                           return BIBlobInfo{std::max(lhs.size, rhs.size), std::max(lhs.alignment, rhs.alignment),
                                             std::max(lhs.owners, rhs.owners)};
                       });

        // Calculate group mappings
        auto &group_mappings = _active_group->mappings();
        int blob_idx = 0;
        for (auto &free_blob: _free_blobs) {
            for (auto &bound_element_id: free_blob.bound_elements) {
                BI_COMPUTE_ERROR_ON(_active_elements.find(bound_element_id) == std::end(_active_elements));
                Element &bound_element = _active_elements[bound_element_id];
                group_mappings[bound_element.handle] = blob_idx;
            }
            ++blob_idx;
        }
    }
}