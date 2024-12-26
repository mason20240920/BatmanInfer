//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_TENSOR_ALLOCATOR_HPP
#define BATMANINFER_BI_TENSOR_ALLOCATOR_HPP

#include <runtime/bi_i_tensor_allocator.hpp>
#include <runtime/bi_memory.hpp>

namespace BatmanInfer {
    class Coordinates;
    class BITensorInfo;

    /**
     * @brief CPU内存张量分配器的基本实现
     */
    class BITensorAllocator : public BIITensorAllocator {

    private:

    };
}

#endif //BATMANINFER_BI_TENSOR_ALLOCATOR_HPP
