//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_BI_I_MEMORY_GROUP_HPP
#define BATMANINFER_BI_I_MEMORY_GROUP_HPP

#include <runtime/bi_memory.hpp>
#include <runtime/bi_types.hpp>

namespace BatmanInfer {
    class BIIMemoryManageable;

    /**
     * @brief 内存组接口
     */
    class BIIMemoryGroup {
    public:
        virtual ~BIIMemoryGroup() = default;

        /**
         * @brief 将一个对象设置为由给定的内存组管理
         *
         * @note 管理者不得最终确定。
         *
         * @param obj
         */
        virtual void manage(BIIMemoryManageable *obj) = 0;

        /**
         * @brief 为给定对象最终确定内存。
         *
         * @note Manager不能被最终确定
         *
         * @param obj 请求内存的对象
         * @param obj_memory 对象的内存处理接口，可用于更改对象使用的底层内存。
         * @param size  内存的大小
         * @param alignment
         */
        virtual void finalize_memory(BIIMemoryManageable *obj,
                                     BIIMemory &obj_memory,
                                     size_t size,
                                     size_t alignment) = 0;

        /**
         * @brief 为整个内存组获取支持内存。
         */
        virtual void acquire() = 0;

        virtual void release() = 0;

        /**
         * @brief 获取该组的内存映射
         * @return
         */
        virtual BIMemoryMappings &mappings() = 0;
    };

    /**
     * @brief 对象能被内存管理的接口
     */
    class BIIMemoryManageable {
    public:
        virtual ~BIIMemoryManageable() = default;

        /**
         * @brief 将一个可管理的内存对象与管理它的内存组关联起来。
         * @param memory_group
         */
        virtual void associate_memory_group(BIIMemoryGroup *memory_group) = 0;
    };

    /**
     * @brief 内存组资源范围处理类
     */
    class BIMemoryGroupResourceScope {
    public:
        explicit BIMemoryGroupResourceScope(BIIMemoryGroup &memory_group) : _memory_group(memory_group)
        {
            _memory_group.acquire();
        }

        ~BIMemoryGroupResourceScope() {
            _memory_group.release();
        }

    private:
        BIIMemoryGroup &_memory_group;
    };
}

#endif //BATMANINFER_BI_I_MEMORY_GROUP_HPP
