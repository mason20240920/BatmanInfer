//
// Created by Mason on 2025/2/11.
//

#pragma once

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;

    namespace cpu {
        void neon_layer_norm_float16(const BIWindow &window,
                                     const BIITensor *input,
                                     BIITensor *output,
                                     const BIITensor *gamma,
                                     const BIITensor *beta,
                                     float epsilon);
    }
}

#endif
