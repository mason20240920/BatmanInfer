//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_I_ALLOCATOR_HPP
#define BATMANINFER_BI_I_ALLOCATOR_HPP

#include <runtime/bi_i_memory_region.hpp>

namespace BatmanInfer {
    /**
     * @brief 内存分配器接口
     */
    class BIIAllocator {
    public:
        virtual ~BIIAllocator() = default;

        /**
         * @brief 由子类实现的接口，用于分配字节。
         * @param size 分配的大小
         * @param alignment 返回的指针应遵循的对齐方式
         * @return 指向已分配内存的指针
         */
        virtual void *allocate(size_t size, size_t alignment) = 0;

        /**
         * @brief 接口由子类实现，以释放分配的张量。
         * @param ptr
         */
        virtual void free(void *ptr) = 0;

        /**
         * @brief 创建自管理内存区域
         * @param size 内存区域的大小
         * @param alignment 内存区域的对齐
         * @return 返回内存区域对象
         */
        virtual std::unique_ptr<BIIMemoryRegion> make_region(size_t size,
                                                             size_t alignment) = 0;
    };
}

#endif //BATMANINFER_BI_I_ALLOCATOR_HPP
