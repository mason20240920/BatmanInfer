//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_BI_I_NE_KERNEL_HPP
#define BATMANINFER_BI_I_NE_KERNEL_HPP

#include <data/core/cpp/bi_i_cpp_kernel.hpp>

namespace BatmanInfer {
    /**
     * @brief 所有在Neon中实现的内核的通用接口。
     */
    using BIINEKernel = BIICPPKernel;
}

#endif //BATMANINFER_BI_I_NE_KERNEL_HPP
