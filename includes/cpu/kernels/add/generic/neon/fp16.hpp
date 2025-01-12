//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_NEON_KERNELS_ADD_FP16_HPP
#define BATMANINFER_NEON_KERNELS_ADD_FP16_HPP

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

namespace BatmanInfer {
    class BIITensor;

    enum class BIConvertPolicy;

    class BIWindow;
    namespace cpu {
        void add_fp16_neon(
                const BIITensor *src0, const BIITensor *src1, BIITensor *dst, const BIConvertPolicy &policy,
                const BIWindow &window);
    } // namespace cpu
} // namespace BatmanInfer
#endif /* (__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */

#endif //BATMANINFER_NEON_KERNELS_ADD_FP16_HPP
