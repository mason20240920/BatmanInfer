//
// Created by Mason on 2025/2/12.
//

#pragma once

#include <cpu/kernels/rms_norm/generic/neon/fp16.hpp>

namespace BatmanInfer {
    namespace cpu {
#define DECLARE_RMS_NORMALIZATION_KERNEL(func_name)                                                      \
    void func_name(const BIWindow *win, const BIITensor *in, const BIITensor *scale, BIITensor *out)

        DECLARE_RMS_NORMALIZATION_KERNEL(neon_rms_norm_fp16_8_impl);

#undef DECLARE_RMS_NORMALIZATION_KERNEL
    }
} // namespace BatmanInfer