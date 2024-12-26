//
// Created by Mason on 2024/12/25.
//

#ifndef BATMANINFER_BI_I_MEMORY_H
#define BATMANINFER_BI_I_MEMORY_H

#include <runtime/bi_i_memory_region.hpp>

namespace BatmanInfer {
    /**
     * @brief 内存接口
     */
    class BIIMemory {
    public:
        /**
         * @brief Virtual default destructor
         */
        virtual ~BIIMemory() = default;

        /**
         * @brief 区域访问器
         * @return 内存区域
         */
        virtual BIIMemoryRegion *region() = 0;

        /**
         * @brief 区域访问器
         * @return 内存区域
         */
        virtual BIIMemoryRegion *region() const = 0;

        /**
         * @brief 设置内存区域
         *
         * @warning 内存区域的所有权归调用者所有
         *
         * @param region 内存区域
         *
         */
        virtual void set_region(BIIMemoryRegion *region) = 0;

        /**
         * @brief 设置内存区域
         *
         * @warning 内存区域的所有权被转移。
         *
         * @param region 内存区域
         */
        virtual void set_owned_region(std::unique_ptr<BIIMemoryRegion> region) = 0;
    };
}


#endif //BATMANINFER_BI_I_MEMORY_H
