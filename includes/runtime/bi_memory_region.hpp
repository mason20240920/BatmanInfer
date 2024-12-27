//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_MEMORY_REGION_HPP
#define BATMANINFER_BI_MEMORY_REGION_HPP

#include <runtime/bi_i_memory_region.hpp>

namespace BatmanInfer {
    class BIMemoryRegion final : public BIIMemoryRegion {
    public:
        /**
         * @brief
         * @param size 区域大小
         * @param alignment 基指针的字节对齐。默认为0。
         */
        explicit BIMemoryRegion(size_t size, size_t alignment = 0) : BIIMemoryRegion(size), _mem(nullptr), _ptr(nullptr)
        {
            if (size != 0) {
                // 分配后端内存
                size_t space = size + alignment;
                /**
                 * @brief
                 * std::shared_ptr<uint8_t>
                 *    C++ 标准库中的一种智能指针
                 *    1. 引用计数来管理动态分配的内存
                 *
                 *  new uint8_t[space]()
                 *    动态分配了一个大小为 space 的字节数组
                 *
                 *  定义了一个自定义删除器，确保在释放内存时使用 delete[] 操作符
                 *   std::shared_ptr<uint8_t>(..., [](uint8_t *ptr) { delete []ptr; })
                 *   这里的 std::shared_ptr 构造函数接受了一个自定义的删除器（deleter）。
                 *   自定义删除器是一个 lambda 表达式：[](uint8_t *ptr) { delete []ptr; }。
                 *   当 std::shared_ptr 的引用计数变为 0 时，会调用这个删除器。
                 *   删除器会使用 delete[] 操作符来释放动态分配的数组内存。
                 *   使用自定义删除器的原因是，std::shared_ptr 默认使用 delete 来释放内存，但这里分配的是数组，所以需要使用 delete[]。
                 */
                _mem = std::shared_ptr<uint8_t>(new uint8_t[space](), [](const uint8_t *ptr) {delete []ptr;});
                _ptr = _mem.get();

                // 计算对齐偏移量
                if (alignment != 0) {
                    void *aligned_ptr = _mem.get();
                    std::align(alignment, size, aligned_ptr, space);
                    _ptr = aligned_ptr;
                }
            }
        }

        BIMemoryRegion(void *ptr, size_t size) : BIIMemoryRegion(size), _mem(nullptr), _ptr(nullptr) {
            if (size != 0)
                _ptr = ptr;
        }

        /**
         * @brief 防止该类的实例被复制（因为该类包含指针）
         */
        BIMemoryRegion(const BIMemoryRegion &) = delete;

        BIMemoryRegion(BIMemoryRegion &&) = default;

        BIMemoryRegion &operator=(const BIMemoryRegion &) = delete;

        BIMemoryRegion &operator=(BIMemoryRegion &&) = default;


        // 继承来自接口函数
        void *buffer() final
        {
            return _ptr;
        }

        const void *buffer() const final {
            return _ptr;
        }

        std::unique_ptr<BIIMemoryRegion> extract_subregion(size_t offset, size_t size) override {
            if (_ptr != nullptr && (offset < _size) && (_size - offset >= size))
                return std::make_unique<BIMemoryRegion>(static_cast<uint8_t *>(_ptr) + offset, size);
            else
                return nullptr;
        }


    protected:
        /**
         * @brief
         */
        std::shared_ptr<uint8_t> _mem;
        void                     *_ptr;

    };
}

#endif //BATMANINFER_BI_MEMORY_REGION_HPP
