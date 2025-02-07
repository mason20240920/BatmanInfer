//
// Created by Mason on 2025/2/7.
//

#pragma once

#include "data/core/bi_helpers.hpp"
#include <cpu/kernels/quantize/generic/neon/fp32.hpp>
#include <cpu/kernels/quantize/generic/neon/fp16.hpp>
#include <cpu/kernels/quantize/generic/neon/integer.hpp>

namespace BatmanInfer {
    namespace cpu {

#define DECLARE_QUANTIZE_KERNEL(func_name) void func_name(const BIITensor *src, BIITensor *dst, const BIWindow &window)

        DECLARE_QUANTIZE_KERNEL(u8_u8_run_quantize_qasymm8);

        DECLARE_QUANTIZE_KERNEL(u8_i8_run_quantize_qasymm8);

        DECLARE_QUANTIZE_KERNEL(i8_u8_run_quantize_qasymm8);

        DECLARE_QUANTIZE_KERNEL(i8_i8_run_quantize_qasymm8);

        DECLARE_QUANTIZE_KERNEL(u8_u8_run_requantize_offset_only);

        DECLARE_QUANTIZE_KERNEL(u8_i8_run_requantize_offset_only);

        DECLARE_QUANTIZE_KERNEL(i8_u8_run_requantize_offset_only);

        DECLARE_QUANTIZE_KERNEL(i8_i8_run_requantize_offset_only);

        DECLARE_QUANTIZE_KERNEL(i8_u8_run_requantize_offset_only_convert);

        DECLARE_QUANTIZE_KERNEL(u8_i8_run_requantize_offset_only_convert);

        DECLARE_QUANTIZE_KERNEL(u8_run_quantize_qasymm16);

        DECLARE_QUANTIZE_KERNEL(i8_run_quantize_qasymm16);

        DECLARE_QUANTIZE_KERNEL(fp32_u8_run_quantize_qasymm8);

        DECLARE_QUANTIZE_KERNEL(fp32_i8_run_quantize_qasymm8);

        DECLARE_QUANTIZE_KERNEL(fp32_run_quantize_qasymm16);

        DECLARE_QUANTIZE_KERNEL(fp32_i8_run_quantize_qsymm8);

        DECLARE_QUANTIZE_KERNEL(fp16_u8_run_quantize_qasymm8);

        DECLARE_QUANTIZE_KERNEL(fp16_i8_run_quantize_qasymm8);

        DECLARE_QUANTIZE_KERNEL(fp16_run_quantize_qasymm16);

#undef DECLARE_QUANTIZE_KERNEL

    } // namespace cpu
} // namespace BatmanInfer