//
// Created by Mason on 2025/1/16.
//

#pragma once

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include <data/core/bi_helpers.hpp>

namespace BatmanInfer {
    namespace cpu {
        template<ArithmeticOperation op>
        void neon_fp16_elementwise_binary(const BIITensor *in1,
                                          const BIITensor *in2,
                                          BIITensor *out,
                                          const BIWindow &window);

        template<ComparisonOperation op>
        void neon_fp16_comparison_elementwise_binary(const BIITensor *in1,
                                                     const BIITensor *in2,
                                                     BIITensor *out,
                                                     const BIWindow &window);
    }
}

#endif