/*
 * Copyright (c) 2022 Arm Limited.
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

#ifndef ARM_COMPUTE_WRAPPER_ERF_H
#define ARM_COMPUTE_WRAPPER_ERF_H

#include <data/core/neon/bi_neon_math.hpp>

#include <neon/neon_defines.h>

namespace BatmanInfer {
    namespace wrapper {
#define VERF_IMPL(vtype, prefix, postfix) \
    inline vtype verf(const vtype &a)     \
    {                                     \
        return prefix##_##postfix(a);     \
    }

        VERF_IMPL(float32x4_t, verfq, f32)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        VERF_IMPL(float16x8_t, verfq, f16)

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VERF_IMPL

    } // namespace wrapper
} // namespace arm_compute

#endif /* ARM_COMPUTE_WRAPPER_ERF_H */
