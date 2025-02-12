//
// Created by Mason on 2025/2/11.
//

#pragma once

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;
    namespace cpu {
        void neon_layer_norm_float16_8_0_2D(
                const BIWindow &window,
                const BIITensor *in,
                const BIITensor *scale,
                const BIITensor *beta,
                BIITensor *out);
    }
}

#endif