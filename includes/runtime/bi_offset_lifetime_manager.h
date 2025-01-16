//
// Created by holynova on 2025/1/15.
//

#pragma once

#include "runtime/bi_i_simple_lifetime_manager.hpp"
#include "runtime/bi_types.hpp"

#include <map>
#include <vector>

namespace BatmanInfer {

    // Forward declarations
    class BIIMemoryPool;

    /** Concrete class that tracks the lifetime of registered tensors and
 *  calculates the systems memory requirements in terms of a single blob and a list of offsets */
    class BIOffsetLifetimeManager : public BIISimpleLifetimeManager
    {
    public:
        using info_type = BIBlobInfo;

    public:
        /** Constructor */
        BIOffsetLifetimeManager();
        /** Prevent instances of this class to be copy constructed */
        BIOffsetLifetimeManager(const BIOffsetLifetimeManager &) = delete;
        /** Prevent instances of this class to be copied */
        BIOffsetLifetimeManager &operator=(const BIOffsetLifetimeManager &) = delete;
        /** Allow instances of this class to be move constructed */
        BIOffsetLifetimeManager(BIOffsetLifetimeManager &&) = default;
        /** Allow instances of this class to be moved */
        BIOffsetLifetimeManager &operator=(BIOffsetLifetimeManager &&) = default;
        /** Accessor to the pool internal configuration meta-data
         *
         * @return Lifetime manager internal configuration meta-data
         */
        const info_type &info() const;

        // Inherited methods overridden:
        std::unique_ptr<BIIMemoryPool> create_pool(BIIAllocator *allocator) override;
        BIMappingType                  mapping_type() const override;

    private:
        // Inherited methods overridden:
        void update_blobs_and_mappings() override;

    private:
        BIBlobInfo _blob; /**< Memory blob size */
    };

} // namespace BatmanInfer
