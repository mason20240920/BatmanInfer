//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_I_TENSOR_ALLOCATOR_H
#define BATMANINFER_BI_I_TENSOR_ALLOCATOR_H

#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>

#include <cstdint>

namespace BatmanInfer {
    class BIITensorAllocator {
    public:
        BIITensorAllocator() = default;

        BIITensorAllocator(const BIITensorAllocator &) = default;

        BIITensorAllocator &operator=(const BIITensorAllocator &) = default;

        BIITensorAllocator(BIITensorAllocator &&) = default;

        BIITensorAllocator &operator=(BIITensorAllocator &&) = default;

        virtual ~BIITensorAllocator() = default;

        /**
         * @brief 根据 @ref BITensorInfo 初始化张量
         *
         *
         * @param input 包含要初始化的张量描述的 TensorInfo 对象。
         * @param alignment 基础指针应遵循的字节对齐。
         */
        void init(const BITensorInfo &input, size_t alignment = 0);

        /**
         * @brief 基于参考 TensorInfo 初始化一个张量
         *
         * @note ITensorAllocator 不会拥有 TensorInfo，因此这些需要存活更久。
         *
         * @param input
         * @param alignment
         */
        void soft_init(BITensorInfo &input, size_t alignment = 0);

        /**
         * @brief 返回张量的元数据引用
         * @return
         */
        BITensorInfo &info();

        const BITensorInfo &info() const;

        /**
         * @brief 返回底层张量缓冲区的对齐方式
         * @return 张量缓存对齐
         */
        size_t alignment() const;

        /**
         * @brief 接口由子类实现以分配张量。
         *
         * @note 预计子类将使用 TensorInfo 来获取内存分配的大小。
         * @warning 张量不能已经被分配。否则调用该函数将会失败。
         */
        virtual void allocate() = 0;

        /**
         * @brief 接口由子类实现以释放已分配的张量
         * @warning 张量已经被分配。否则调用该函数将会失败。
         */
        virtual void free() = 0;

        /**
         * @brief 返回张量当前是否已分配。
         * @return
         */
        virtual bool is_allocated() const = 0;

    protected:
        /**
         * @brief 接口由子类实现，以锁定CPU访问的内存分配。
         * @return
         */
        virtual uint8_t *lock() = 0;

        /**
         * @brief 接口由子类实现，以在CPU完成访问后解锁内存分配。
         */
        virtual void unlock() = 0;

    private:
        /**
         * @brief 张量的元数据
         * @return
         */
        BITensorInfo _info_owned() {};

        /**
         * @brief 额外的张量信息
         */
        BITensorInfo *_info_external{nullptr};

        /**
         * @brief 张量在字节中的对齐
         */
        size_t _alignment{};
    };
}

#endif //BATMANINFER_BI_I_TENSOR_ALLOCATOR_H
