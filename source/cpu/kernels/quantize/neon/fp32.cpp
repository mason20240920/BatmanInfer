//
// Created by Mason on 2025/2/7.
//

#include <cpu/kernels/quantize/generic/neon/impl.hpp>
#include <cpu/kernels/quantize/generic/neon/fp32.hpp>

namespace BatmanInfer {
    namespace cpu {
        void fp32_u8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm8<float, uint8_t>(src, dst, window);
        }

        void fp32_i8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm8<float, int8_t>(src, dst, window);
        }

        void fp32_run_quantize_qasymm16(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm16<float>(src, dst, window);
        }

        void fp32_i8_run_quantize_qsymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qsymm8<float, int8_t>(src, dst, window);
        }
    } // namespace cpu
} // namespace BatmanInfer
