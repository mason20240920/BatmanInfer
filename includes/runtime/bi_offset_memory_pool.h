//
// Created by holynova on 2025/1/15.
//

#pragma once

#include "runtime/bi_i_memory_pool.hpp"
#include "runtime/bi_i_memory_region.hpp"
#include "runtime/bi_types.hpp"

#include <cstddef>
#include <memory>

namespace BatmanInfer {

    // Forward declarations
    class BIIAllocator;

    /** Offset based memory pool */
    class BIOffsetMemoryPool : public BIIMemoryPool
    {
    public:
        /** Default Constructor
         *
         * @note allocator should outlive the memory pool
         *
         * @param[in] allocator Backing memory allocator
         * @param[in] blob_info Configuration information of the blob to be allocated
         */
        BIOffsetMemoryPool(BIIAllocator *allocator, BIBlobInfo blob_info);
        /** Default Destructor */
        ~BIOffsetMemoryPool() = default;
        /** Prevent instances of this class to be copy constructed */
        BIOffsetMemoryPool(const BIOffsetMemoryPool &) = delete;
        /** Prevent instances of this class to be copy assigned */
        BIOffsetMemoryPool &operator=(const BIOffsetMemoryPool &) = delete;
        /** Allow instances of this class to be move constructed */
        BIOffsetMemoryPool(BIOffsetMemoryPool &&) = default;
        /** Allow instances of this class to be move assigned */
        BIOffsetMemoryPool &operator=(BIOffsetMemoryPool &&) = default;
        /** Accessor to the pool internal configuration meta-data
         *
         * @return Pool internal configuration meta-data
         */
        const BIBlobInfo &info() const;

        // Inherited methods overridden:
        void                           acquire(BIMemoryMappings &handles) override;
        void                           release(BIMemoryMappings &handles) override;
        BIMappingType                  mapping_type() const override;
        std::unique_ptr<BIIMemoryPool> duplicate() override;

    private:
        BIIAllocator                    *_allocator; /**< Allocator to use for internal allocation */
        std::unique_ptr<BIIMemoryRegion> _blob;      /**< Memory blob */
        BIBlobInfo                       _blob_info; /**< Information for the blob to allocate */
    };

} // namespace BatmanInfer
