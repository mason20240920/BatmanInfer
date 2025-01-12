//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_CPU_KERNELS_ADD_LIST_HPP
#define BATMANINFER_CPU_KERNELS_ADD_LIST_HPP

#include <cpu/kernels/add/generic/neon/fp16.hpp>
#include <cpu/kernels/add/generic/neon/fp32.hpp>
#include <cpu/kernels/add/generic/neon/integer.hpp>
#include <cpu/kernels/add/generic/neon/qasymm8.hpp>
#include <cpu/kernels/add/generic/neon/qasymm8_signed.hpp>
#include <cpu/kernels/add/generic/neon/qasymm16.hpp>
#include <cpu/kernels/add/generic/neon/impl.hpp>

namespace BatmanInfer {

    namespace cpu {
#define DECLARE_ADD_KERNEL(func_name)                                                                   \
     void func_name(const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy, \
                   const BIWindow &window)

        DECLARE_ADD_KERNEL(add_qasymm8_neon);

        DECLARE_ADD_KERNEL(add_qasymm8_signed_neon);

        DECLARE_ADD_KERNEL(add_qsymm16_neon);

        DECLARE_ADD_KERNEL(add_fp32_neon);

        DECLARE_ADD_KERNEL(add_fp16_neon);

        DECLARE_ADD_KERNEL(add_u8_neon);

        DECLARE_ADD_KERNEL(add_s16_neon);

        DECLARE_ADD_KERNEL(add_s32_neon);

#undef DECLARE_ADD_KERNEL

    }
}

#endif //BATMANINFER_CPU_KERNELS_ADD_LIST_HPP
