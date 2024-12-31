//
// Created by Mason on 2024/12/30.
//

#ifndef BATMANINFER_BI_POOL_MANAGER_HPP
#define BATMANINFER_BI_POOL_MANAGER_HPP

#include <runtime/bi_i_memory_pool.hpp>
#include <runtime/i_pool_manager.hpp>

#include <support/mutex.hpp>
#include <support/semaphore.hpp>

#include <list>

namespace BatmanInfer {
    class BIPoolManager : public BIIPoolManager {
    public:
        BIPoolManager();

        BIPoolManager(const BIPoolManager &) = delete;

        BIPoolManager &operator=(const BIPoolManager &) = delete;

        BIPoolManager(BIPoolManager &&) = delete;

        BIPoolManager &operator=(BIPoolManager &&) = delete;

        // 继承的类复写
        BIIMemoryPool * lock_pool() override;
        void unlock_pool(BatmanInfer::BIIMemoryPool *pool) override;
        void register_pool(std::unique_ptr<BIIMemoryPool> pool) override;
        std::unique_ptr<BIIMemoryPool> release_pool() override;
        void clear_pools() override;
        size_t num_pools() const override;

    private:
        /**
         * @brief 空闲的内存池
         */
        std::list<std::unique_ptr<BIIMemoryPool>> _free_pools;

        /**
         * @brief 占用的内存池
         */
        std::list<std::unique_ptr<BIIMemoryPool>> _occupied_pools;

        /**
         * @brief 控制队列的信号量
         */
        std::unique_ptr<Semaphore> _sem;

        /**
         * @brief 互斥锁控制去队列的权限
         */
        mutable Mutex _mtx;
    };
}

#endif //BATMANINFER_BI_POOL_MANAGER_HPP
