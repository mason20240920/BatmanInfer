//
// Created by Mason on 2025/2/11.
//

#include <cpu/kernels/layer_norm/generic/neon/fp16.hpp>
#include <cpu/kernels/layer_norm/generic/neon/impl.hpp>
#include <cpu/bi_cpu_types.hpp>

namespace BatmanInfer {
    namespace cpu {
        void neon_layer_norm_float16_8_0_2D(
                const BIWindow &window,
                const BIITensor *in,
                const BIITensor *scale,
                const BIITensor *beta,
                BIITensor *out) {
            layer_norm<float16_t, 8, true>(window, in, out, scale, beta, 1e-5f);
        }
    }
}