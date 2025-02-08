//
// Created by Mason on 2025/2/8.
//

#include <cpu/kernels/dequantize/generic/neon/fp32.hpp>
#include <cpu/kernels/dequantize/generic/neon/impl.hpp>

namespace BatmanInfer {
    namespace cpu {
        void fp32_run_dequantization_core(const BIITensor *input, BIITensor *output, const BIWindow &window) {
            run_dequantization_core<float>(input, output, window);
        }
    }
}