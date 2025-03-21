/*
 * Copyright (c) 2018-2020, 2022 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
        /**
         * 定义一个宏 `VSTORE_IMPL`，用于生成通用的 SIMD 数据存储函数。
         * - `stype`：标量数据类型（如 uint8_t、int16_t）。
         * - `vtype`：向量数据类型（如 uint8x8_t、float32x4_t）。
         * - `prefix`：NEON 存储指令的前缀（如 vst1, vst1q）。
         * - `postfix`：NEON 存储指令的后缀，表示数据类型（如 u8, s16）。
         *
         * 宏展开后生成一个内联函数 `vstore`，用于调用对应的 NEON 指令，
         * 将向量数据（`val`）存储到指针 `ptr` 指向的内存中。
         */
#define VSTORE_IMPL(stype, vtype, prefix, postfix) \
    inline void vstore(stype *ptr, vtype val)      \
    {                                              \
        prefix##_##postfix(ptr, val);              \
    }

        VSTORE_IMPL(uint8_t, uint8x8_t, vst1, u8)

        VSTORE_IMPL(uint8_t, uint8x8x2_t, vst2, u8)

        VSTORE_IMPL(int8_t, int8x8_t, vst1, s8)

        VSTORE_IMPL(int8_t, int8x8x2_t, vst2, s8)

        VSTORE_IMPL(uint16_t, uint16x4_t, vst1, u16)

        VSTORE_IMPL(int16_t, int16x4_t, vst1, s16)

        VSTORE_IMPL(uint32_t, uint32x2_t, vst1, u32)

        VSTORE_IMPL(int32_t, int32x2_t, vst1, s32)

        VSTORE_IMPL(float, float32x2_t, vst1, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VSTORE_IMPL(float16_t, float16x4_t, vst1, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VSTORE_IMPL(uint8_t, uint8x16_t, vst1q, u8)

        VSTORE_IMPL(int8_t, int8x16_t, vst1q, s8)

        VSTORE_IMPL(uint16_t, uint16x8_t, vst1q, u16)

        VSTORE_IMPL(int16_t, int16x8_t, vst1q, s16)

        VSTORE_IMPL(uint32_t, uint32x4_t, vst1q, u32)

        VSTORE_IMPL(int32_t, int32x4_t, vst1q, s32)

        VSTORE_IMPL(float, float32x4_t, vst1q, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VSTORE_IMPL(float16_t, float16x8_t, vst1q, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VSTORE_IMPL
    } // namespace wrapper
} // namespace BatmanInfer
