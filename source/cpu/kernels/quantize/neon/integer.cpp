//
// Created by Mason on 2025/2/7.
//

#include <cpu/kernels/quantize/generic/neon/integer.hpp>
#include <cpu/kernels/quantize/generic/neon/impl.hpp>

namespace BatmanInfer {
    namespace cpu {
        void u8_u8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm8<uint8_t, uint8_t>(src, dst, window);
        }

        void u8_i8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm8<uint8_t, int8_t>(src, dst, window);
        }

        void i8_u8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm8<int8_t, uint8_t>(src, dst, window);
        }

        void i8_i8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm8<int8_t, int8_t>(src, dst, window);
        }

        void u8_run_quantize_qasymm16(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm16<uint8_t>(src, dst, window);
        }

        void i8_run_quantize_qasymm16(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_quantize_qasymm16<int8_t>(src, dst, window);
        }

        void u8_u8_run_requantize_offset_only(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_requantize_offset_only<uint8_t, uint8_t>(src, dst, window);
        }

        void u8_i8_run_requantize_offset_only(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_requantize_offset_only<uint8_t, int8_t>(src, dst, window);
        }

        void i8_u8_run_requantize_offset_only(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_requantize_offset_only<int8_t, uint8_t>(src, dst, window);
        }

        void i8_i8_run_requantize_offset_only(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_requantize_offset_only<int8_t, int8_t>(src, dst, window);
        }

        void i8_u8_run_requantize_offset_only_convert(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_requantize_offset_only_convert<int8_t, uint8_t>(src, dst, window);
        }

        void u8_i8_run_requantize_offset_only_convert(const BIITensor *src, BIITensor *dst, const BIWindow &window) {
            run_requantize_offset_only_convert<uint8_t, int8_t>(src, dst, window);
        }
    } // namespace cpu
} // namespace BatmanInfer