//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_BI_I_LIFETIME_MANAGER_HPP
#define BATMANINFER_BI_I_LIFETIME_MANAGER_HPP

#include <runtime/bi_i_memory_pool.hpp>
#include <runtime/bi_types.hpp>

namespace BatmanInfer {
    class BIIAllocator;
    class BIIMemory;
    class BIIMemoryGroup;

    /**
     * @brief 管理对象生命周期的接口
     */
    class BIILifetimeManager {
    public:
        virtual ~BIILifetimeManager() = default;

        /**
         * @brief 注册一个内存组到生命周期管理器，并且赋予一个组id
         * @param group 内存组id
         */
        virtual void register_group(BIIMemoryGroup *group) = 0;

        /**
         * @brief 释放与一个内存组相关的元素
         * @param group
         */
        virtual void release_group(BIIMemoryGroup *group) = 0;

        /**
         * @brief 注册并开始对象的生命周期
         * @param obj
         */
        virtual void start_lifetime(void *obj) = 0;

        /**
         * @brief 一个对象的生命周期结束
         * @param obj 对象
         * @param obj_memory 对象内存
         * @param size 对象的大小
         * @param alignment 对象的对齐要求
         */
        virtual void end_life_time(void *obj,
                                   BIIMemory &obj_memory,
                                   size_t size,
                                   size_t alignment) = 0;

        /**
         * @brief 检查注册对象的生命周期是否已完成。
         * @return
         */
        virtual bool are_all_finalized() const = 0;

        /**
         * @brief 根据内存需求创建内存池。
         * @param allocator
         * @return
         */
        virtual std::unique_ptr<BIIMemoryPool> create_pool(BIIAllocator *allocator) = 0;

        /**
         * @brief 返回生命周期管理器返回的映射类型
         * @return
         */
        virtual BIMappingType mapping_type() const = 0;
    };
}

#endif //BATMANINFER_BI_I_LIFETIME_MANAGER_HPP
