//
// Created by Mason on 2025/2/12.
//

#include <cpu/kernels/rms_norm/generic/neon/impl.hpp>
#include <cpu/kernels/rms_norm/generic/neon/fp16.hpp>

namespace BatmanInfer {
    namespace cpu {
        void neon_rms_norm_fp16_8_impl(
            const BIWindow &window,
            const BIITensor *in,
            const BIITensor *scale,
            const BIITensor *out) {
            rms_norm_fp16(window, in, out, scale);
        }
    }
}
