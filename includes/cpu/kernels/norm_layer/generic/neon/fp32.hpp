//
// Created by Mason on 2025/2/9.
//

#pragma once

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;

    class BINormalizationLayerInfo;
    namespace cpu {
        void neon_normalize_float32_4_0_2D(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);

        void neon_normalize_float32_4_0(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);

        void neon_normalize_float32_4_1_2D(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);

        void neon_normalize_float32_4_1(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);

        void neon_normalize_float32_4_2(
                const BIWindow &window, const BIITensor *in, const BIITensor *in_squared, BIITensor *out,
                BINormalizationLayerInfo ninfo);
    }
}