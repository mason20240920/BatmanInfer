/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_QMOVUN_H
#define ARM_COMPUTE_WRAPPER_QMOVUN_H

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VQMOVUN_IMPL(dtype, vtype, prefix, postfix) \
    inline dtype vqmovun(const vtype &a)            \
    {                                               \
        return prefix##_##postfix(a);               \
    }

        VQMOVUN_IMPL(uint32x2_t, int64x2_t, vqmovun, s64)

        VQMOVUN_IMPL(uint16x4_t, int32x4_t, vqmovun, s32)

        VQMOVUN_IMPL(uint8x8_t, int16x8_t, vqmovun, s16)

#undef VQMOVUN_IMPL
    } // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_QMOVUN_H */
