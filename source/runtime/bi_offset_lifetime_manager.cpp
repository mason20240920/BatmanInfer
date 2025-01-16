//
// Created by holynova on 2025/1/15.
//

#include "runtime/bi_offset_lifetime_manager.h"

#include "data/core/bi_error.h"
#include "runtime/bi_i_allocator.hpp"
#include "runtime/bi_i_memory_group.hpp"
#include "runtime/bi_offset_memory_pool.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace BatmanInfer {

    namespace {

    	size_t align_offset(size_t offset, size_t alignment)
        {
            const size_t remainder = (alignment != 0U) ? offset % alignment : 0U;
            return (remainder != 0U) ? offset + (alignment - remainder) : offset;
        }

    } // namespace unnamed

    BIOffsetLifetimeManager::BIOffsetLifetimeManager() : _blob(0)
    {
    }

    const BIOffsetLifetimeManager::info_type &BIOffsetLifetimeManager::info() const
    {
        return _blob;
    }

    std::unique_ptr<BIIMemoryPool> BIOffsetLifetimeManager::create_pool(BIIAllocator *allocator)
    {
        BI_COMPUTE_ERROR_ON(allocator == nullptr);
        return std::make_unique<BIOffsetMemoryPool>(allocator, _blob);
    }

    BIMappingType BIOffsetLifetimeManager::mapping_type() const
    {
        return BIMappingType::OFFSETS;
    }

    void BIOffsetLifetimeManager::update_blobs_and_mappings()
    {
        BI_COMPUTE_ERROR_ON(!are_all_finalized());
        BI_COMPUTE_ERROR_ON(_active_group == nullptr);

        // Update blob size
        size_t max_aggregated_size = 0;
        std::for_each(std::begin(_free_blobs), std::end(_free_blobs),
                      [&](const Blob &b)
                      {
                          max_aggregated_size += b.max_size;
                          _blob.alignment = std::max(_blob.alignment, b.max_alignment);
                      });
        max_aggregated_size += _free_blobs.size() * _blob.alignment;
        _blob.owners = std::max(_blob.owners, _free_blobs.size());
        _blob.size   = std::max(_blob.size, max_aggregated_size);

        // Calculate group mappings
        auto  &group_mappings = _active_group->mappings();
        size_t offset         = 0;
        for (auto &free_blob : _free_blobs)
        {
            for (auto &bound_element_id : free_blob.bound_elements)
            {
                BI_COMPUTE_ERROR_ON(_active_elements.find(bound_element_id) == std::end(_active_elements));
                Element &bound_element               = _active_elements[bound_element_id];
                group_mappings[bound_element.handle] = offset;
            }
            offset += free_blob.max_size;
            offset = align_offset(offset, _blob.alignment);
            BI_COMPUTE_ERROR_ON(offset > _blob.size);
        }
	}

} // namespace BatmanInfer
