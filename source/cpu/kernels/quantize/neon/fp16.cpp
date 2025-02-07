//
// Created by Mason on 2025/2/7.
//

#include <cpu/kernels/quantize/generic/neon/impl.hpp>
#include <cpu/kernels/quantize/generic/neon/fp16.hpp>

namespace BatmanInfer {
    namespace cpu {
        void fp16_u8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm8<float16_t, uint8_t>(src, dst, window);
        }

        void fp16_i8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm8<float16_t, int8_t>(src, dst, window);
        }

        void fp16_run_quantize_qasymm16(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm16<float16_t>(src, dst, window);
        }
    }
}