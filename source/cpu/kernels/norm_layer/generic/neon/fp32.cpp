//
// Created by Mason on 2025/2/9.
//

#include <cpu/kernels/norm_layer/generic/neon/impl.hpp>
#include <cpu/kernels/norm_layer/generic/neon/fp32.hpp>
#include <cpu/bi_cpu_types.hpp>

namespace BatmanInfer {
    namespace cpu {
        void neon_normalize_float32_4_0_2D(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo) {
            normalize_float<float, 4, 0, true>(window, in, in_squared, out, ninfo);
        }

        void neon_normalize_float32_4_0(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo) {
            normalize_float<float, 4, 0, false>(window, in, in_squared, out, ninfo);
        }

        void neon_normalize_float32_4_1_2D(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo) {
            normalize_float<float, 4, 1, true>(window, in, in_squared, out, ninfo);
        }

        void neon_normalize_float32_4_1(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo) {
            normalize_float<float, 4, 1, false>(window, in, in_squared, out, ninfo);
        }

        void neon_normalize_float32_4_2(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo) {
            normalize_float<float, 4, 2, false>(window, in, in_squared, out, ninfo);
        }
    }
}