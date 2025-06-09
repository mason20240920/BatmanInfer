//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_TENSOR_ALLOCATOR_HPP
#define BATMANINFER_BI_TENSOR_ALLOCATOR_HPP

#include <runtime/bi_i_tensor_allocator.hpp>
#include <runtime/bi_memory.hpp>
#include <runtime/bi_memory_group.hpp>

#include <data/core/bi_error.h>

namespace BatmanInfer {
    class BICoordinates;
    class BITensorInfo;

    /**
     * @brief CPU内存张量分配器的基本实现
     */
    class BITensorAllocator : public BIITensorAllocator {
    public:
        /**
         * @brief 内存管理的拥有者
         * @param owner
         */
        BITensorAllocator(BIIMemoryManageable *owner);

        ~BITensorAllocator();

        BITensorAllocator(const BITensorAllocator &) = delete;

        BITensorAllocator &operator=(const BITensorAllocator &) = delete;

        BITensorAllocator(BITensorAllocator &&) noexcept;

        BITensorAllocator &operator=(BITensorAllocator &&) noexcept;

        /**
         * @brief 使初始化方法可用
         */
        using BIITensorAllocator::init;

        /**
         * @brief 与另一个张量分配器共享相同的底层内存，但张量信息可能不同
         *        换句话说，这可以用来从另一个张量创建一个子张量，同时共享相同的内存
         *
         * @note 张量分配器拥有相同的类型
         *
         * @param allocator 拥有待共享后备内存的分配器。所有权随后变为共享。
         * @param coords 新张量在父张量内的起始坐标。
         * @param sub_info 新的张量信息（例如：形状等）
         */
        void init(const BITensorAllocator &allocator,
                  const BICoordinates &coords,
                  BITensorInfo &sub_info);

        /**
         * @brief 与另一个张量分配器共享相同的底层内存，但张量信息可能不同
         *        换句话说，这可以用来从另一个张量创建一个子张量，同时共享相同的内存
         * @warning 但是与上面的init不同的是，这个内存是连续的，不需要修改BIStrides为父向量的BIStrides
         * @param allocator 拥有待共享后备内存的分配器。所有权随后变为共享
         * @param sub_info 新的张量信息（例如：形状等）
         */
        void init(const BITensorAllocator &allocator,
                  BITensorInfo &sub_info);

        /**
         * @brief 返回指向已分配数据的指针。
         *
         * @return 指针指向分配的数据
         */
        uint8_t *data() const;

        uint8_t *data(const size_t offset, const size_t size) const;

        /**
         * @brief 分配由 TensorInfo 指定大小的 CPU 内存。
         *
         * @note 调用此函数时，张量必须已经被分配。
         */
        void allocate() override;

        bool is_allocated() const override;

        /**
         * @brief 释放CPU内存
         */
        void free() override;

        /**
         * @brief 将现有内存导入为张量的底层内存
         *
         * @warning 内存的大小应与 ITensorInfo 报告的 total_size 保持一致。
         * @warning 内存的所有权不会被转移。
         * @warning 张量不应由内存管理器管理。
         * @warning 填充（padding）需要由客户端代码处理。
         * @warning 如果存在原地（in-place）操作，内存必须是可写的。
         *
         * @note 将检查缓冲区的对齐方式，以确保与 ITensorInfo 报告的对齐要求一致。
         *
         * @param memory 用作底层内存的原始内存指针
         * @return 错误状态
         */
        BIStatus import_memory(void *memory);

        /**
         * @brief 将张量与一个内存组关联起来
         *
         * @param associated_memory_group 与张量关联的内存组
         */
        void set_associated_memory_group(BIIMemoryGroup *associated_memory_group);

    protected:
        /**
         * @brief 对 CPU 内存无操作
         *
         * @return 一个指向张量分配起始位置的指针。
         */
        uint8_t *lock() override;

        /**
         * @brief 对CPU内存无操作
         */
        void unlock() override;

    private:
        /**
         * @brief 内存管理对象: 拥有分配器
         */
        BIIMemoryManageable *_owner;

        /**
         * @brief 注册内存管理器
         */
        BIIMemoryGroup *_associated_memory_group;

        /**
         * @brief CPU内存
         */
        BIMemory _memory;
    };
}

#endif //BATMANINFER_BI_TENSOR_ALLOCATOR_HPP
