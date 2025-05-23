/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_EOR_H
#define ARM_COMPUTE_WRAPPER_EOR_H

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VEOR_IMPL(vtype, prefix, postfix)             \
    inline vtype veor(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

        VEOR_IMPL(uint8x8_t, veor, u8)

        VEOR_IMPL(int8x8_t, veor, s8)

        VEOR_IMPL(uint16x4_t, veor, u16)

        VEOR_IMPL(int16x4_t, veor, s16)

        VEOR_IMPL(uint32x2_t, veor, u32)

        VEOR_IMPL(int32x2_t, veor, s32)

        VEOR_IMPL(uint8x16_t, veorq, u8)

        VEOR_IMPL(int8x16_t, veorq, s8)

        VEOR_IMPL(uint16x8_t, veorq, u16)

        VEOR_IMPL(int16x8_t, veorq, s16)

        VEOR_IMPL(uint32x4_t, veorq, u32)

        VEOR_IMPL(int32x4_t, veorq, s32)

#undef VEOR_IMPL
    } // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_EOR_H */
