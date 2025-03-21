/*
 * Copyright (c) 2017-2021, 2024 Arm Limited.
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

#ifdef __aarch64__

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>

#include "../performance_parameters.hpp"
#include "../std_transforms_fixed.hpp"

namespace BatmanGemm {

// Load the actual kernel
    void a64_gemm_s8_8x12(const int8_t *, const int8_t *, int32_t *, int, int, int);

    void a64_gemm_s8_8x12_a55r1(const int8_t *, const int8_t *, int32_t *, int, int, int);

    void a64_gemm_s8_8x12_x1(const int8_t *, const int8_t *, int32_t *, int, int, int);

    class cls_a64_gemm_s8_8x12 {
    public:
        typedef int8_t lhs_operand_type;
        typedef int8_t rhs_operand_type;
        typedef int32_t result_type;

        typedef void (*kern_type)(const int8_t *, const int8_t *, int32_t *, int, int, int);

        /* Kernel blocking parameters */
        static unsigned int out_width() {
            return 12;
        }

        static unsigned int out_height() {
            return 8;
        }

        static unsigned int k_unroll() {
            return 4;
        }

        // Use the standard fixed size transforms.
        StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 4> transforms = {};
        StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 8, 12, 4, true> transforms_quantized = {};

        template<typename T>
        static PerformanceParameters get_performance_parameters(const BICPUInfo *ci) {
            if (std::is_same<T, int8_t>::value) {
                switch (ci->get_cpu_model()) {
                    case BICPUModel::A510:
                        return {19.73, 3.38, 0.27};

                    case BICPUModel::A55r1:
                        return {15.361, 0.9341, 0.1636};

                    case BICPUModel::V1:
                        return {51.14, 7.38, 0.65};

                    default:
                        return {29.0698, 3.9793, 0.4003};
                }
            }

            if (std::is_same<T, int32_t>::value) {
                switch (ci->get_cpu_model()) {
                    case BICPUModel::A510:
                        return {19.73, 3.38, 3.70};

                    case BICPUModel::A55r1:
                        return {14.286, 1.171, 1.209};

                    case BICPUModel::V1:
                        return {61.58, 4.78, 10.83};

                    default:
                        return {31.82, 3.51, 8.03};
                }
            }
        }

        kern_type kernel = a64_gemm_s8_8x12;

        cls_a64_gemm_s8_8x12(const BICPUInfo *ci) {
            auto mod = ci->get_cpu_model();

            if (mod == BICPUModel::A55r1) {
                kernel = a64_gemm_s8_8x12_a55r1;
            } else if (mod == BICPUModel::X1) {
                kernel = a64_gemm_s8_8x12_x1;
            }
        }
    };

} // namespace arm_gemm

#endif // __aarch64__
