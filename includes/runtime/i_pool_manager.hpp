//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_I_POOL_MANAGER_HPP
#define BATMANINFER_I_POOL_MANAGER_HPP

#include <memory>

namespace BatmanInfer {
    class BIIMemoryPool;

    /**
     * @brief 内存池管理接口
     */
    class BIIPoolManager {
    public:
        virtual ~BIIPoolManager() = default;

        /**
         * @brief 锁上内存池来进行执行程序
         * @return  锁定的池将映射工作负载。
         */
        virtual BIIMemoryPool *lock_pool() = 0;

        /**
         * @brief 释放内存池
         * @param pool 需要释放的内存池
         */
        virtual void unlock_pool(BIIMemoryPool *pool) = 0;

        /**
         * @brief 注册池由池管理
         *
         * @note 内存池的所有权正在转移给内存池管理者。
         *
         * @param pool 要被管理的内存池
         */
        virtual void register_pool(std::unique_ptr<BIIMemoryPool> pool) = 0;

        /**
         * @brief 从管理的内存池释放一个 free 的内存池
         * @return 被释放的内存池，否则为nullptr
         */
        virtual std::unique_ptr<BIIMemoryPool> release_pool() = 0;

        /**
         * @brief 清除所有由内存池管理器管理的池。
         *
         * @pre 所有的内存池必须没有被占用
         *
         */
        virtual void clear_pools() = 0;

        /**
         * @brief 返回被内存管理器管理的内存池总数
         * @return 管理的内存池的数量
         */
        virtual size_t num_pools() const = 0;
    };
}

#endif //BATMANINFER_I_POOL_MANAGER_HPP
