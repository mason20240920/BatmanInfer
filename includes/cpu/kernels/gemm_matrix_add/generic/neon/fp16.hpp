//
// Created by Mason on 2025/1/8.
//

#ifndef BATMANINFER_MATRIX_ADD__FP16_HPP
#define BATMANINFER_MATRIX_ADD__FP16_HPP

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    class BIWindow;
    namespace cpu {
        void neon_fp16_gemm_matrix_add(const BIITensor *src,
                                       BIITensor *dst,
                                       const BIWindow &window,
                                       float beta);
    }
}

#endif

#endif //BATMANINFER_MATRIX_ADD__FP16_HPP
