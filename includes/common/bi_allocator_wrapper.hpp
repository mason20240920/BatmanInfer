//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_BI_ALLOCATOR_WRAPPER_HPP
#define BATMANINFER_BI_ALLOCATOR_WRAPPER_HPP

#include "bcl_types.hpp"

namespace BatmanInfer {
    /**
     * 内存分配器实现
     */
    class BIAllocatorWrapper final {
    public:
        /**
         * 默认构造函数
         * @param backing_allocator 后端内存被使用
         */
        BIAllocatorWrapper(const BclAllocator &backing_allocator) noexcept;

        BIAllocatorWrapper(const BIAllocatorWrapper &) noexcept = default;

        BIAllocatorWrapper(BIAllocatorWrapper &&) noexcept = default;

        BIAllocatorWrapper &operator=(const BIAllocatorWrapper &) noexcept = delete;

        BIAllocatorWrapper &operator=(BIAllocatorWrapper &&) noexcept = default;

        /**
         * 分配一块指定大小（以字节为单位）的内存
         * @param size
         * @return
         */
        void *alloc(size_t size);

        /**
         * 释放一个已经分配的内存
         * @param ptr
         */
        void free(void *ptr);

        /**
         * 分配指定大小（以字节为单位）的内存块，同时满足给定的对齐要求
         *
         * @param size 要分配的呢库存大小（以字节为单位）
         * @param alignment
         * @return
         */
        void *aligned_alloc(size_t size,
                            size_t alignment);

        /**
         * 释放一个对齐的内存块
         * @param ptr
         */
        void aligned_free(void *ptr);

        /**
         * 根据分配器设置用户数据
         * @param user_data
         */
        void set_user_data(void *user_data);

    private:
        BclAllocator _backing_allocator;
    };
}

#endif //BATMANINFER_BI_ALLOCATOR_WRAPPER_HPP
