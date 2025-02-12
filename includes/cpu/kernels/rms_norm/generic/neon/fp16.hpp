//
// Created by Mason on 2025/2/12.
//

#pragma once

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

namespace BatmanInfer {
    class BIITensor;

    namespace cpu {
        /**
         * float16x8_t 方式对维度进行归一化
         * @param in
         * @param scale
         * @param out
         */
        void neon_rms_norm_fp16_8_impl(
                const BIITensor *in,
                const BIITensor *scale,
                BIITensor *out);
    }
}

#endif