//
// Created by Mason on 2024/12/27.
//

#include "runtime/bi_memory_manager_on_demand.hpp"

#include <data/core/bi_error.h>
#include <runtime/bi_blob_lifetime_manager.hpp>
//#include <runtime/pool>

namespace BatmanInfer {
    BIMemoryManagerOnDemand::BIMemoryManagerOnDemand(std::shared_ptr<BIILifetimeManager> lifetime_manager,
                                                     std::shared_ptr<BIIPoolManager> pool_manager):
                                                     _lifetime_mgr(std::move(lifetime_manager)),
                                                     _pool_mgr(std::move(pool_manager)){
        BI_COMPUTE_ERROR_ON_MSG(!_lifetime_mgr, "Lifetime manager not specified correctly!");
        BI_COMPUTE_ERROR_ON_MSG(!_pool_mgr, "Pool manager not specified correctly!");
    }

    BIILifetimeManager *BIMemoryManagerOnDemand::lifetime_manager() {
        return _lifetime_mgr.get();
    }

    BIIPoolManager *BIMemoryManagerOnDemand::pool_manager() {
        return _pool_mgr.get();
    }

    void BIMemoryManagerOnDemand::populate(BatmanInfer::BIIAllocator &allocator,
                                            size_t num_pools) {
        BI_COMPUTE_ERROR_ON(!_lifetime_mgr);
        BI_COMPUTE_ERROR_ON(!_pool_mgr);
        BI_COMPUTE_ERROR_ON_MSG(!_lifetime_mgr->are_all_finalized(),
                                "All the objects have not been finalized!");
        BI_COMPUTE_ERROR_ON_MSG(_pool_mgr->num_pools() != 0,
                                "Pool manager already contains pools!");

        // Create pools
        auto pool_template = _lifetime_mgr->create_pool(&allocator);
        for (int i = num_pools; i > 1; --i) {
            auto pool = pool_template->duplicate();
            _pool_mgr->register_pool(std::move(pool));
        }
        _pool_mgr->register_pool(std::move(pool_template));
    }

    void BIMemoryManagerOnDemand::clear() {
        BI_COMPUTE_ERROR_ON_MSG(!_pool_mgr, "Pool manager not specified correctly!");
        _pool_mgr->clear_pools();
    }

    std::shared_ptr<BIMemoryManagerOnDemand> BIMemoryManagerOnDemand::make_default() {
        auto lifetime_mgr = std::make_shared<BIBlobLifetimeManager>();
//        auto pool_mgr = std::make_shared<>()
    }
}
