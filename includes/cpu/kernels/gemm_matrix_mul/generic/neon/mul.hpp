//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_MATRIX_MUL_MUL_HPP
#define BATMANINFER_MATRIX_MUL_MUL_HPP

#include <data/core/bi_helpers.hpp>

#include <data/core/cpp/bi_cpp_validate.hpp>

namespace BatmanInfer {
    namespace cpu {
        void vector_matrix_multiply_f32(
                const BIITensor *lhs,
                const BIITensor *rhs,
                BIITensor *dst,
                const BIWindow &window,
                const ThreadInfo &info,
                float alpha);

        void matrix_matrix_multiply_f32(
                const BIITensor *lhs,
                const BIITensor *rhs,
                BIITensor *dst,
                const BIWindow &window,
                const ThreadInfo &info,
                float alpha);

    } // namespace cpu
}

#endif //BATMANINFER_MATRIX_MUL_MUL_HPP
