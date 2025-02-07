//
// Created by Mason on 2025/2/6.
//

#pragma once

namespace BatmanInfer {
    class BIITensor;

    class BIWindow;

    class ThreadInfo;
    namespace cpu {
        void neon_s8_gemm_matrix_mul(const BIITensor *lhs,
                                     const BIITensor *rhs,
                                     BIITensor *dst,
                                     const BIWindow &window,
                                     const ThreadInfo &info,
                                     float alpha,
                                     const bool is_dst_vector);
    }
}