//
// Created by Mason on 2025/2/8.
//

#pragma once

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;
    namespace cpu {
        void fp16_run_dequantization_core(const BIITensor *input, BIITensor *output, const BIWindow &window);
    } // namespace cpu
} // namespace BatmanInfer
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */