//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_MATRIX_MUL_FP16_HPP
#define BATMANINFER_MATRIX_MUL_FP16_HPP

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;

    class ThreadInfo;
    namespace cpu {
        void neon_fp16_gemm_matrix_mul(const BIITensor *lhs,
                                       const BIITensor *rhs,
                                       BIITensor *dst,
                                       const BIWindow &window,
                                       const ThreadInfo &info,
                                       float alpha,
                                       const bool is_dst_vector);
    }
}

#endif //BATMANINFER_MATRIX_MUL_FP16_HPP