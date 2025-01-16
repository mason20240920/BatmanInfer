//
// Created by holynova on 2025/1/15.
//

#include "runtime/bi_offset_memory_pool.h"

#include "data/core/bi_error.h"
#include "runtime/bi_i_allocator.hpp"
#include "runtime/bi_i_memory_pool.hpp"
#include "runtime/bi_memory_region.hpp"
#include "runtime/bi_types.hpp"

#include <algorithm>

namespace BatmanInfer {

	BIOffsetMemoryPool::BIOffsetMemoryPool(BIIAllocator *allocator, BIBlobInfo blob_info)
        : _allocator(allocator), _blob(), _blob_info(blob_info)
    {
        BI_COMPUTE_ERROR_ON(!allocator);
        _blob = _allocator->make_region(blob_info.size, blob_info.alignment);
    }

    const BIBlobInfo &BIOffsetMemoryPool::info() const
    {
        return _blob_info;
    }

    void BIOffsetMemoryPool::acquire(BIMemoryMappings &handles)
    {
        BI_COMPUTE_ERROR_ON(_blob == nullptr);

        // Set memory to handlers
        for (auto &handle : handles)
        {
            BI_COMPUTE_ERROR_ON(handle.first == nullptr);
            handle.first->set_owned_region(_blob->extract_subregion(handle.second, _blob_info.size - handle.second));
        }
    }

    void BIOffsetMemoryPool::release(BIMemoryMappings &handles)
    {
        for (auto &handle : handles)
        {
            BI_COMPUTE_ERROR_ON(handle.first == nullptr);
            handle.first->set_region(nullptr);
        }
    }

    BIMappingType BIOffsetMemoryPool::mapping_type() const
    {
        return BIMappingType::OFFSETS;
    }

    std::unique_ptr<BIIMemoryPool> BIOffsetMemoryPool::duplicate()
    {
        BI_COMPUTE_ERROR_ON(!_allocator);
        return std::make_unique<BIOffsetMemoryPool>(_allocator, _blob_info);
    }

} // namespace BatmanInfer
