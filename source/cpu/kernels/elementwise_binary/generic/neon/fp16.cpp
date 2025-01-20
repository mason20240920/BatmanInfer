//
// Created by Mason on 2025/1/16.
//

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

//#include "arm_compute/core/Helpers.h"

#include "cpu/kernels/elementwise_binary/generic/neon/fp16.hpp"

#include "cpu/kernels/elementwise_binary/generic/neon/impl.hpp"

namespace BatmanInfer {
    namespace cpu {
        template<ArithmeticOperation op>
        void neon_fp16_elementwise_binary(const BIITensor *in1, const BIITensor *in2, BIITensor *out,
                                          const BIWindow &window) {
            return elementwise_arithm_op<op, typename wrapper::traits::neon_vector<float16_t, 8>>(in1, in2, out,
                                                                                                  window);
        }

        template void neon_fp16_elementwise_binary<ArithmeticOperation::ADD>(const BIITensor *in1,
                                                                             const BIITensor *in2,
                                                                             BIITensor *out,
                                                                             const BIWindow &window);

        template void neon_fp16_elementwise_binary<ArithmeticOperation::SUB>(const BIITensor *in1,
                                                                             const BIITensor *in2,
                                                                             BIITensor *out,
                                                                             const BIWindow &window);

        template void neon_fp16_elementwise_binary<ArithmeticOperation::DIV>(const BIITensor *in1,
                                                                             const BIITensor *in2,
                                                                             BIITensor *out,
                                                                             const BIWindow &window);

        template void neon_fp16_elementwise_binary<ArithmeticOperation::MIN>(const BIITensor *in1,
                                                                             const BIITensor *in2,
                                                                             BIITensor *out,
                                                                             const BIWindow &window);

        template void neon_fp16_elementwise_binary<ArithmeticOperation::MAX>(const BIITensor *in1,
                                                                             const BIITensor *in2,
                                                                             BIITensor *out,
                                                                             const BIWindow &window);

        template void neon_fp16_elementwise_binary<ArithmeticOperation::SQUARED_DIFF>(const BIITensor *in1,
                                                                                      const BIITensor *in2,
                                                                                      BIITensor *out,
                                                                                      const BIWindow &window);

        template void neon_fp16_elementwise_binary<ArithmeticOperation::POWER>(const BIITensor *in1,
                                                                               const BIITensor *in2,
                                                                               BIITensor *out,
                                                                               const BIWindow &window);

        template void neon_fp16_elementwise_binary<ArithmeticOperation::PRELU>(const BIITensor *in1,
                                                                               const BIITensor *in2,
                                                                               BIITensor *out,
                                                                               const BIWindow &window);

        template<ComparisonOperation op>
        void neon_fp16_comparison_elementwise_binary(const BIITensor *in1, const BIITensor *in2, BIITensor *out,
                                                     const BIWindow &window) {
            return elementwise_comp_op_16<op, float16_t, float16x8_t>(in1, in2, out, window);
        }

        template void neon_fp16_comparison_elementwise_binary<ComparisonOperation::Equal>(const BIITensor *in1,
                                                                                          const BIITensor *in2,
                                                                                          BIITensor *out,
                                                                                          const BIWindow &window);

        template void neon_fp16_comparison_elementwise_binary<ComparisonOperation::NotEqual>(const BIITensor *in1,
                                                                                             const BIITensor *in2,
                                                                                             BIITensor *out,
                                                                                             const BIWindow &window);

        template void neon_fp16_comparison_elementwise_binary<ComparisonOperation::Greater>(const BIITensor *in1,
                                                                                            const BIITensor *in2,
                                                                                            BIITensor *out,
                                                                                            const BIWindow &window);

        template void neon_fp16_comparison_elementwise_binary<ComparisonOperation::GreaterEqual>(const BIITensor *in1,
                                                                                                 const BIITensor *in2,
                                                                                                 BIITensor *out,
                                                                                                 const BIWindow &window);

        template void neon_fp16_comparison_elementwise_binary<ComparisonOperation::Less>(const BIITensor *in1,
                                                                                         const BIITensor *in2,
                                                                                         BIITensor *out,
                                                                                         const BIWindow &window);

        template void neon_fp16_comparison_elementwise_binary<ComparisonOperation::LessEqual>(const BIITensor *in1,
                                                                                              const BIITensor *in2,
                                                                                              BIITensor *out,
                                                                                              const BIWindow &window);
    } // namespace cpu
} // namespace BatmanInfer
#endif //defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)