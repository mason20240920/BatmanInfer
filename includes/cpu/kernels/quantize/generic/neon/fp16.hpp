//
// Created by Mason on 2025/2/7.
//

#pragma once

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;
    namespace cpu {
        void fp16_u8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void fp16_i8_run_quantize_qasymm8(const BIITensor *src, BIITensor *dst, const BIWindow &window);

        void fp16_run_quantize_qasymm16(const BIITensor *src, BIITensor *dst, const BIWindow &window);
    } // namespace cpu
} // namespace BatmanInfer
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */