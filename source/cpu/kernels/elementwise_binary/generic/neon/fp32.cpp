//
// Created by Mason on 2025/1/16.
//

#include "cpu/kernels/elementwise_binary/generic/neon/fp32.hpp"
#include "cpu/kernels/elementwise_binary/generic/neon/impl.hpp"

namespace BatmanInfer {
    namespace cpu {
        template<ArithmeticOperation op>
        void neon_fp32_elementwise_binary(const BIITensor *in1, const BIITensor *in2, BIITensor *out,
                                          const BIWindow &window) {
            return elementwise_arithm_op<op, typename wrapper::traits::neon_vector<float, 4>>(in1, in2, out, window);
        }

        template void neon_fp32_elementwise_binary<ArithmeticOperation::ADD>(const BIITensor *in1,
                                                                             const BIITensor *in2,
                                                                             BIITensor *out,
                                                                             const BIWindow &window);

        template void neon_fp32_elementwise_binary<ArithmeticOperation::SUB>(const BIITensor *in1,
                                                                             const BIITensor *in2,
                                                                             BIITensor *out,
                                                                             const BIWindow &window);

        template void neon_fp32_elementwise_binary<ArithmeticOperation::DIV>(const BIITensor *in1,
                                                                             const BIITensor *in2,
                                                                             BIITensor *out,
                                                                             const BIWindow &window);

        template void neon_fp32_elementwise_binary<ArithmeticOperation::MIN>(const BIITensor *in1,
                                                                             const BIITensor *in2,
                                                                             BIITensor *out,
                                                                             const BIWindow &window);

        template void neon_fp32_elementwise_binary<ArithmeticOperation::MAX>(const BIITensor *in1,
                                                                             const BIITensor *in2,
                                                                             BIITensor *out,
                                                                             const BIWindow &window);

        template void neon_fp32_elementwise_binary<ArithmeticOperation::SQUARED_DIFF>(const BIITensor *in1,
                                                                                      const BIITensor *in2,
                                                                                      BIITensor *out,
                                                                                      const BIWindow &window);

        template void neon_fp32_elementwise_binary<ArithmeticOperation::POWER>(const BIITensor *in1,
                                                                               const BIITensor *in2,
                                                                               BIITensor *out,
                                                                               const BIWindow &window);

        template void neon_fp32_elementwise_binary<ArithmeticOperation::PRELU>(const BIITensor *in1,
                                                                               const BIITensor *in2,
                                                                               BIITensor *out,
                                                                               const BIWindow &window);

        template<ComparisonOperation op>
        void neon_fp32_comparison_elementwise_binary(const BIITensor *in1, const BIITensor *in2, BIITensor *out,
                                                     const BIWindow &window) {
            return elementwise_comp_op_32<op, float, float32x4_t>(in1, in2, out, window);
        }

        template void neon_fp32_comparison_elementwise_binary<ComparisonOperation::Equal>(const BIITensor *in1,
                                                                                          const BIITensor *in2,
                                                                                          BIITensor *out,
                                                                                          const BIWindow &window);

        template void neon_fp32_comparison_elementwise_binary<ComparisonOperation::NotEqual>(const BIITensor *in1,
                                                                                             const BIITensor *in2,
                                                                                             BIITensor *out,
                                                                                             const BIWindow &window);

        template void neon_fp32_comparison_elementwise_binary<ComparisonOperation::Greater>(const BIITensor *in1,
                                                                                            const BIITensor *in2,
                                                                                            BIITensor *out,
                                                                                            const BIWindow &window);

        template void neon_fp32_comparison_elementwise_binary<ComparisonOperation::GreaterEqual>(const BIITensor *in1,
                                                                                                 const BIITensor *in2,
                                                                                                 BIITensor *out,
                                                                                                 const BIWindow &window);

        template void neon_fp32_comparison_elementwise_binary<ComparisonOperation::Less>(const BIITensor *in1,
                                                                                         const BIITensor *in2,
                                                                                         BIITensor *out,
                                                                                         const BIWindow &window);

        template void neon_fp32_comparison_elementwise_binary<ComparisonOperation::LessEqual>(const BIITensor *in1,
                                                                                              const BIITensor *in2,
                                                                                              BIITensor *out,
                                                                                              const BIWindow &window);
    }
}