//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_BI_I_MEMORY_MANAGER_HPP
#define BATMANINFER_BI_I_MEMORY_MANAGER_HPP

#include <runtime/bi_i_lifetime_manager.hpp>
#include <runtime/i_pool_manager.hpp>

namespace BatmanInfer {
    /**
     * @brief 内存管理器接口用于处理后备内存的分配。
     */
    class BIIMemoryManager {
    public:
        virtual ~BIIMemoryManager() = default;

        /**
         * @brief 返回内存管理器使用的生命周期管理器
         * @return
         */
        virtual BIILifetimeManager *lifetime_manager() = 0;

        /**
         * @brief 返回内存管理器使用的池管理器
         * @return 内存池管理器
         */
        virtual BIIPoolManager *pool_manager() = 0;

        /**
         * @brief 根据给定的内存池数量生成内存池管理器
         * @param allocator 分配器: 分配后端资源的分配器
         * @param num_pools 内存池数量
         */
        virtual void populate(BIIAllocator &allocator, size_t num_pools) = 0;

        /**
         * @brief 清空内存管理器
         *
         * @pre 所有的内存池必须没有被占用
         */
        virtual void clear() = 0;
    };
}

#endif //BATMANINFER_BI_I_MEMORY_MANAGER_HPP
