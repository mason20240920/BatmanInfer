/*
 * Copyright (c) 2024 Arm Limited.
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

#include "../std_transforms_fixed.hpp"
#include "../bfloat.hpp"
#include "../kernel_weight_format.hpp"
#include "../performance_parameters.hpp"

#define ARGLIST  \
    unsigned int, const unsigned int *, \
    IndirectInputArg<float>, \
    size_t, size_t, \
    const bfloat16 *, \
    size_t, \
    IndirectOutputArg<float>, \
    const float *, Activation, bool

namespace BatmanGemm {
// Actual kernel implementations
    void a64_ffhybrid_fp32bf16fp32_mmla_6x16(ARGLIST);

    class cls_a64_ffhybrid_fp32bf16fp32_mmla_6x16 {
    public:
        typedef float lhs_operand_type;
        typedef bfloat16 rhs_operand_type;
        typedef float result_type;

        typedef void (*kern_type)(ARGLIST);

        /* Kernel blocking parameters */
        static constexpr unsigned int out_height() {
            return 6;
        }

        static unsigned int stripe_width() {
            return 4;
        }

        static KernelWeightFormat kernel_weight_format() {
            return KernelWeightFormat::VL256_BL64_BF16;
        }

        static unsigned int out_width() {
            return 16;
        }

        static constexpr unsigned int k_unroll() {
            return 4;
        }

        static constexpr bool supports_accumulate() {
            return true;
        }

        StdTransformsFixed<lhs_operand_type, rhs_operand_type, result_type, 6, 16, 4> transforms = {};

        template<typename T>
        static inline PerformanceParameters get_performance_parameters(const BICPUInfo *ci) {
            if (std::is_same<T, float>::value) {
                switch (ci->get_cpu_model()) {
                    case BICPUModel::V1:
                        return {21.05};
                    default:
                        return {15.27};
                }
            }

            return {1.0};
        }

        // Default to the generic kernel
        kern_type kernel = a64_ffhybrid_fp32bf16fp32_mmla_6x16;

        cls_a64_ffhybrid_fp32bf16fp32_mmla_6x16(const BICPUInfo *) {
        }
    };

} // namespace arm_gemm

#undef ARGLIST
#endif // __aarch64__
