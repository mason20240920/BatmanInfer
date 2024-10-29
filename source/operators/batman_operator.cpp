//
// Created by Mason on 2024/10/28.
//

#include <operators/batman_operator.hpp>
#include "omp.h"

namespace BatmanInfer {
    void BatmanOperator::csr_matrix_vector_multiply(const int *row_ptr,
                                                    const int *col_indices,
                                                    const double *values,
                                                    const double *x,
                                                    double *y,
                                                    int num_rows) {
        #pragma omp parallel for
        for (int i = 0; i < num_rows; ++i) {
            double sum = 0.0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                sum += values[j] * x[col_indices[j]];
            }
            y[i] = sum;
        }
    }

    void BatmanOperator::dense_matrix_vector_multiply(const std::vector<std::vector<double>>& dense_matrix,
                                      const double *x,
                                      double *y,
                                      int num_rows) {
        #pragma omp parallel for
        for (int i = 0; i < num_rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < num_rows; ++j) {
                sum += dense_matrix[i][j] * x[j];
            }
            y[i] = sum;
        }
    }
}