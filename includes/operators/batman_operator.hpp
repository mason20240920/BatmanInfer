//
// Created by Mason on 2024/10/28.
//

#ifndef BATMAN_INFER_BATMAN_OPERATOR_HPP
#define BATMAN_INFER_BATMAN_OPERATOR_HPP
#include <vector>

namespace BatmanInfer {
    class BatmanOperator {
    public:
        /**
         * 矩阵进行计算
         * @param row_ptr
         * @param col_indices
         * @param values
         * @param x
         * @param y
         * @param num_rows
         */
        static void csr_matrix_vector_multiply(const int *row_ptr,
                                               const int *col_indices,
                                               const double *values,
                                               const double *x,
                                               double *y,
                                               int num_rows);

        /**
         * 稠密矩阵计算
         * @param dense_matrix
         * @param x
         * @param y
         * @param num_rows
         */
        static void dense_matrix_vector_multiply(const std::vector<std::vector<double>> &dense_matrix,
                                                 const double *x,
                                                 double *y,
                                                 int num_rows);
    };
}

#endif //BATMAN_INFER_BATMAN_OPERATOR_HPP
