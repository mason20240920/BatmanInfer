/*
 * Copyright (c) 2021, 2023-2024 Arm Limited.
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

#include "../std_transform_fixed_trB.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<float>, \
    size_t, size_t, \
    const float *, \
    IndirectOutputArg<float>, \
    const float *, Activation, bool

namespace BatmanGemm {
// Actual kernel implementations
    void a64_hybrid_fp32_mla_4x24(ARGLIST);

    void a64_hybrid_fp32_mla_4x24_a55(ARGLIST);

    class cls_a64_hybrid_fp32_mla_4x24 {
    public:
        typedef float lhs_operand_type;
        typedef float rhs_operand_type;
        typedef float result_type;

        typedef void (*kern_type)(ARGLIST);

        /* Kernel blocking parameters */
        static constexpr unsigned int out_height() {
            return 4;
        }

        static unsigned int out_width() {
            return 24;
        }

        static constexpr unsigned int k_unroll() {
            return 1;
        }

        static constexpr bool supports_accumulate() {
            return true;
        }

        StdTransformsFixedTRB<lhs_operand_type, rhs_operand_type, result_type, 4, 24, 1> transforms = {};

        template<typename T>
        static inline PerformanceParameters get_performance_parameters(const BICPUInfo *ci) {
            if (std::is_same<T, float>::value) {
                switch (ci->get_cpu_model()) {
                    case BICPUModel::A55r1:
                        return {2.985};
                    case BICPUModel::A53:
                        return {1.43};
                    case BICPUModel::A73:
                        return {2.56};
                    case BICPUModel::A510:
                        return {3.51};
                    case BICPUModel::V1:
                        return {13.86};
                    default:
                        return {6.614};
                }
            }

            return {1.0};
        }

        // Default to the generic kernel
        kern_type kernel = a64_hybrid_fp32_mla_4x24;

        cls_a64_hybrid_fp32_mla_4x24(const BICPUInfo *ci) {
            switch (ci->get_cpu_model()) {
                default:
                    break;
                case BICPUModel::A55r1:
                case BICPUModel::A53:
                    kernel = a64_hybrid_fp32_mla_4x24_a55;
                    break;
            }
        }
    };

} // namespace arm_gemm

#undef ARGLIST
#endif // __aarch64__
