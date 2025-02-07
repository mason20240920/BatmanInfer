//
// Created by Mason on 2025/2/7.
//

#pragma once

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;
    namespace cpu {
        void u8_u8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void u8_i8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void i8_u8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void i8_i8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void u8_run_quantize_qasymm16(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void i8_run_quantize_qasymm16(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void u8_u8_run_requantize_offset_only(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void u8_i8_run_requantize_offset_only(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void i8_u8_run_requantize_offset_only(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void i8_i8_run_requantize_offset_only(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void i8_u8_run_requantize_offset_only_convert(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void u8_i8_run_requantize_offset_only_convert(const BIITensor *src, BIITensor *dst, const BIWindow &window);
    } // namespace cpu
} // namespace BatmanInfer