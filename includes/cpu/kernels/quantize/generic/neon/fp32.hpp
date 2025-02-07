//
// Created by Mason on 2025/2/7.
//

#pragma once

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;
    namespace cpu {
        void fp32_u8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void fp32_i8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void fp32_run_quantize_qasymm16(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void fp32_i8_run_quantize_qsymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window);
    } // namespace cpu
} // namespace BatmanInfer