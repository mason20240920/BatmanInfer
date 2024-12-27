//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_BI_I_MEMORY_POOL_HPP
#define BATMANINFER_BI_I_MEMORY_POOL_HPP

#include <runtime/bi_types.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace BatmanInfer {
    /**
     * @brief 内存池接口
     */
    class BIIMemoryPool {
    public:
        virtual ~BIIMemoryPool() = default;

        /**
         * @brief 将占用者设置到内存池
         * @param handles 一组成对的向量（句柄，索引）
         */
        virtual void acquire(BIMemoryMappings &handles) = 0;

        /**
         * @brief 释放一块内存块
         * @param handles 一个包含一对句柄和索引的向量
         */
        virtual void release(BIMemoryMappings &handles) = 0;

        /**
         * @brief 返回该池接受的映射类型
         * @return
         */
        virtual BIMappingType mapping_type() const = 0;

        /**
         * @brief 拷贝已经存在的内存池
         * @return
         */
        virtual std::unique_ptr<BIIMemoryPool> duplicate() = 0;
    };
}

#endif //BATMANINFER_BI_I_MEMORY_POOL_HPP
