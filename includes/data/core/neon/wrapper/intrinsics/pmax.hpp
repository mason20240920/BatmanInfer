/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_PMAX_H
#define ARM_COMPUTE_WRAPPER_PMAX_H

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VPMAX_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vpmax(const vtype &a, const vtype &b) \
    {                                                  \
        return prefix##_##postfix(a, b);               \
    }

        VPMAX_IMPL(uint8_t, uint8x8_t, vpmax, u8)

        VPMAX_IMPL(int8_t, int8x8_t, vpmax, s8)

        VPMAX_IMPL(uint16_t, uint16x4_t, vpmax, u16)

        VPMAX_IMPL(int16_t, int16x4_t, vpmax, s16)

        VPMAX_IMPL(uint32_t, uint32x2_t, vpmax, u32)

        VPMAX_IMPL(int32_t, int32x2_t, vpmax, s32)

        VPMAX_IMPL(float, float32x2_t, vpmax, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VPMAX_IMPL(float16_t, float16x4_t, vpmax, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VPMAX_IMPL
    } // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_PMAX_H */
