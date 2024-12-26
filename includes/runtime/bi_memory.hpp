//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_MEMORY_HPP
#define BATMANINFER_BI_MEMORY_HPP

#include <runtime/bi_i_memory.hpp>
#include <runtime/bi_i_memory_region.hpp>

#include <cstddef>
#include <memory>

namespace BatmanInfer {
    /**
     * @brief CPU内存对象实现
     */
    class BIMemory : public BIIMemory {
    public:
        BIMemory();

        /**
         * @brief 默认构造函数
         * @param memory 需要被导入的内存
         */
        explicit BIMemory(const std::shared_ptr<BIIMemoryRegion> &memory);

        /**
         * @brief 默认构造函数
         *
         * @note 内存的所有权不会转移到这个对象上。
         *       因此，管理（分配/释放）应该由客户端来完成。
         *
         * @param memory 需要被导入的内存
         */
        explicit BIMemory(BIIMemoryRegion *memory);

        BIMemory(const BIMemory &) = default;

        BIMemory &operator=(const BIMemory &) = default;

        BIMemory(BIMemory &&) noexcept = default;

        BIMemory &operator=(BIMemory &&) noexcept = default;

        // 接口方法的视线
        BIIMemoryRegion *region() final;

        BIIMemoryRegion *region() const final;

        void set_region(BIIMemoryRegion *region) final;

        void set_owned_region(std::unique_ptr<BIIMemoryRegion> region) final;


    private:
        /**
         * @brief 内存区域
         */
        BIIMemoryRegion                  *_region;
        /**
         * @brief 内存区域拥有者
         */
        std::shared_ptr<BIIMemoryRegion> _region_owned;
    };
}

#endif //BATMANINFER_BI_MEMORY_HPP
