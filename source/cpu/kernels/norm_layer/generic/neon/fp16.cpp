//
// Created by Mason on 2025/2/9.
//
#include <cpu/kernels/norm_layer/generic/neon/impl.hpp>
#include <cpu/kernels/norm_layer/generic/neon/fp16.hpp>
#include <cpu/bi_cpu_types.hpp>

namespace BatmanInfer {
    namespace cpu {
        void neon_normalize_float16_8_0_2D(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo) {
            normalize_float<float16_t, 8, 0, true>(window, in, in_squared, out, ninfo);
        }

        void neon_normalize_float16_8_0(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo) {
            normalize_float<float16_t, 8, 0, false>(window, in, in_squared, out, ninfo);
        }

        void neon_normalize_float16_8_1_2D(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo) {
            normalize_float<float16_t, 8, 1, true>(window, in, in_squared, out, ninfo);
        }

        void neon_normalize_float16_8_1(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo) {
            normalize_float<float16_t, 8, 1, false>(window, in, in_squared, out, ninfo);
        }

        void neon_normalize_float16_8_2(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo) {
            normalize_float<float16_t, 8, 2, false>(window, in, in_squared, out, ninfo);
        }
    }
}