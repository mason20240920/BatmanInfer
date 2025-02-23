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
#ifndef ARM_COMPUTE_WRAPPER_SIN_H
#define ARM_COMPUTE_WRAPPER_SIN_H

#include <data/core/neon/bi_neon_math.hpp>

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VSIN_IMPL(vtype, prefix, postfix) \
    inline vtype vsin(const vtype &a)     \
    {                                     \
        return prefix##_##postfix(a);     \
    }

#define VSIN_IMPL_INT(vtype, prefix, postfix) \
    inline vtype vsin(const vtype &a)         \
    {                                         \
        BI_COMPUTE_UNUSED(a);                \
        BI_COMPUTE_ERROR("Not supported");   \
    }

        VSIN_IMPL(float32x4_t, vsinq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VSIN_IMPL(float16x8_t, vsinq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VSIN_IMPL_INT(int32x4_t, vsinq, s32)

#undef vsub_IMPL
    } // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_SUB_H */
