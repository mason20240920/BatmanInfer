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
#ifndef ARM_COMPUTE_WRAPPER_QMOV_H
#define ARM_COMPUTE_WRAPPER_QMOV_H

#include <neon/neon_defines.h>
#include <type_traits>


namespace BatmanInfer {
    namespace wrapper {
        template<typename T>
        inline typename std::enable_if<std::is_same<T, uint8_t>::value, uint8x8_t>::type vqmov(const int16x8_t &a) {
            return vqmovun_s16(a);
        }

        template<typename T>
        inline typename std::enable_if<std::is_same<T, int8_t>::value, int8x8_t>::type vqmov(const int16x8_t &a) {
            return vqmovn_s16(a);
        }

    } // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_QMOV_H */
