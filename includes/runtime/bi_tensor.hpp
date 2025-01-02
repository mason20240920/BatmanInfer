//
// Created by Mason on 2024/12/31.
//

#ifndef BATMANINFER_BI_TENSOR_HPP
#define BATMANINFER_BI_TENSOR_HPP

#include <data/core/bi_i_tensor.hpp>
#include <runtime/bi_tensor_allocator.hpp>

namespace BatmanInfer {
    class BIITensor;
    class BIIRuntimeContext;

    class BITensor: public BIITensor, public BIIMemoryManageable{
    public:
        /**
         * @brief 初始化函数
         * @param ctx (可选) 指向运行的上下文内容
         */
        BITensor(BIIRuntimeContext *ctx = nullptr);

        /**
         * @brief 释放Tensor的内存
         */
        ~BITensor() = default;

        /**
         * @brief 允许默认的右移函数
         */
        BITensor(BITensor &&) = default;

        BITensor & operator=(BITensor &&) = default;

        /**
         * @brief 返回指向张量分配器的指针
         *
         * @return 张量分配器的指针
         */
        BITensorAllocator *allocator();

        // 继承于重写函数
        BIITensorInfo * info() const override;
        BIITensorInfo * info() override;
        uint8_t * buffer() const override;
        void associate_memory_group(BIIMemoryGroup *memory_group) override;

    private:
        /**
         * @brief 基本的CPU分配器的实例
         */
        mutable BITensorAllocator _allocator;
    };

    using Image = BITensor;
}

#endif //BATMANINFER_BI_TENSOR_HPP
