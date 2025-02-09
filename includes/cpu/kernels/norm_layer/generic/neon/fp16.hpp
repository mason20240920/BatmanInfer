//
// Created by Mason on 2025/2/9.
//

#pragma once

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;

    class BINormalizationLayerInfo;
    namespace cpu {
        void neon_normalize_float16_8_0_2D(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);

        void neon_normalize_float16_8_0(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);

        void neon_normalize_float16_8_1_2D(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);

        void neon_normalize_float16_8_1(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);

        void neon_normalize_float16_8_2(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);
    }
}

#endif