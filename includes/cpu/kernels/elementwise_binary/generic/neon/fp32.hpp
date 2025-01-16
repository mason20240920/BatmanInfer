//
// Created by Mason on 2025/1/16.
//

#pragma once

#include <data/core/bi_helpers.hpp>

namespace BatmanInfer {
    namespace cpu {
        template<ArithmeticOperation op>
        void neon_fp32_elementwise_binary(const BIITensor *in1,
                                          const BIITensor *in2,
                                          BIITensor *out,
                                          const BIWindow &window);

        template<ComparisonOperation op>
        void neon_fp32_comparison_elementwise_binary(const BIITensor *in1,
                                                     const BIITensor *in2,
                                                     BIITensor *out,
                                                     const BIWindow &window);
    }
}