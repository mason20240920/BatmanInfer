//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_BI_MEMORY_MANAGER_ON_DEMAND_HPP
#define BATMANINFER_BI_MEMORY_MANAGER_ON_DEMAND_HPP

#include <runtime/bi_i_lifetime_manager.hpp>
#include <runtime/bi_i_memory_group.hpp>
#include <runtime/bi_i_memory_manager.hpp>
#include <runtime/i_pool_manager.hpp>

namespace BatmanInfer {
    /**
     * @brief 按需内存管理器
     */
    class BIMemoryManagerOnDemand : public BIIMemoryManager {
    public:
        BIMemoryManagerOnDemand(std::shared_ptr<BIILifetimeManager> lifetime_manager,
                                std::shared_ptr<BIIPoolManager> pool_manager);

        /**
         * @brief Prevent instances of this class to be copy constructed
         */
        BIMemoryManagerOnDemand(const BIMemoryManagerOnDemand &) = delete;

        BIMemoryManagerOnDemand &operator=(const BIMemoryManagerOnDemand &) = delete;

        BIMemoryManagerOnDemand(BIMemoryManagerOnDemand &&) = default;

        BIMemoryManagerOnDemand &operator=(BIMemoryManagerOnDemand &&) = default;

        static std::shared_ptr<BIMemoryManagerOnDemand> make_default();

        // 继承方法
        BIILifetimeManager *lifetime_manager() override;
        BIIPoolManager * pool_manager() override;
        void populate(BatmanInfer::BIIAllocator &allocator, size_t num_pools) override;
        void clear() override;

    private:
        /**
         * @brief 生命中期管理器
         */
        std::shared_ptr<BIILifetimeManager> _lifetime_mgr;

        /**
         * @brief 内存池管理器
         */
        std::shared_ptr<BIIPoolManager> _pool_mgr;
    };
}


#endif //BATMANINFER_BI_MEMORY_MANAGER_ON_DEMAND_HPP
