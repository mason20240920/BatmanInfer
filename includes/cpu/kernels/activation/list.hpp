//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_CPU_KERNELS_ACTIVATION_LIST_HPP
#define BATMANINFER_CPU_KERNELS_ACTIVATION_LIST_HPP

#include <cpu/kernels/activation/generic/neon/lut.hpp>

#include <cpu/kernels/activation/generic/neon/qasymm8.hpp>

#include <cpu/kernels/activation/generic/neon/qasymm8_signed.hpp>

#include <cpu/kernels/activation/generic/neon/qsymm16.hpp>

#include <cpu/kernels/activation/generic/neon/fp16.hpp>

#include <cpu/kernels/activation/generic/neon/fp32.hpp>

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_ACTIVATION_KERNEL(func_name) \
    void func_name(const BIITensor *src, BIITensor *dst, const BIActivationLayerInfo &act_info, const BIWindow &window)

#ifdef __aarch64__

        DECLARE_ACTIVATION_KERNEL(neon_q8_activation_lut);

#endif // __aarch64__

//        DECLARE_ACTIVATION_KERNEL(sve2_q8_activation_lut);

        DECLARE_ACTIVATION_KERNEL(neon_qasymm8_activation);

//        DECLARE_ACTIVATION_KERNEL(sve2_qasymm8_activation);

        DECLARE_ACTIVATION_KERNEL(neon_qasymm8_signed_activation);

//        DECLARE_ACTIVATION_KERNEL(sve2_qasymm8_signed_activation);

        DECLARE_ACTIVATION_KERNEL(neon_qsymm16_activation);

//        DECLARE_ACTIVATION_KERNEL(sve2_qsymm16_activation);

//        DECLARE_ACTIVATION_KERNEL(sve_fp16_activation);

//        DECLARE_ACTIVATION_KERNEL(sve_fp16_activation_lut);

//        DECLARE_ACTIVATION_KERNEL(sve_fp32_activation);

        DECLARE_ACTIVATION_KERNEL(neon_fp16_activation);

        DECLARE_ACTIVATION_KERNEL(neon_fp32_activation);

#undef DECLARE_ACTIVATION_KERNEL
    }
}

#endif //BATMANINFER_CPU_KERNELS_ACTIVATION_LIST_HPP
