//
// Created by Mason on 2025/2/11.
//

#include <cpu/kernels/layer_norm/generic/neon/fp16.hpp>
#include <cpu/kernels/layer_norm/generic/neon/impl.hpp>
#include <cpu/bi_cpu_types.hpp>

namespace BatmanInfer {
    namespace cpu {
        void neon_layer_norm_float16(const BIWindow &window,
                                     const BIITensor *input,
                                     BIITensor *output,
                                     const BIITensor *gamma,
                                     const BIITensor *beta,
                                     float epsilon) {
            layer_norm_fp16(window,
                            input,
                            output,
                            gamma,
                            beta,
                            epsilon);
        }
    }
}