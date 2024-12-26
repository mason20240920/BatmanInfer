//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_MEMORY_REGION_HPP
#define BATMANINFER_BI_MEMORY_REGION_HPP

#include <runtime/bi_i_memory_region.hpp>

namespace BatmanInfer {
    class BIMemoryRegion final : public BIIMemoryRegion {
        /**
         * @brief
         * @param size 区域大小
         * @param alignment 基指针的字节对齐。默认为0。
         */
        BIMemoryRegion(size_t size, size_t alignment = 0) : BIIMemoryRegion(size), _mem(nullptr), _ptr(nullptr)
        {
            if (size != 0) {

            }
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
