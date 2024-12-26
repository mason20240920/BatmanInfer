//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_ALLOCATOR_HPP
#define BATMANINFER_BI_ALLOCATOR_HPP

#include <runtime/bi_i_allocator.hpp>
#include <runtime/bi_i_memory_region.hpp>

namespace BatmanInfer {
    /**
     * @brief 默认的 malloc 分配器实现
     */
    class BIAllocator final : public BIIAllocator {
    public:
        /**
         * @brief 默认构造函数
         */
        BIAllocator() = default;

        // 接口继承
        void * allocate(size_t size, size_t alignment) override;
        void free(void *ptr) override;
        std::unique_ptr<BIIMemoryRegion> make_region(size_t size, size_t alignment) override;
    };
}

#endif //BATMANINFER_BI_ALLOCATOR_HPP
