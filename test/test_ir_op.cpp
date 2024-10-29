//
// Created by Mason on 2024/10/28.
//

#include <iostream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <fstream>
#include <onnx/onnx_pb.h>
#include <google/protobuf/text_format.h>
#include <others/utils.hpp>
#include <runtime/ir.h>
#include <operators/batman_operator.hpp>

/**
 * 计算CSR稀疏矩阵的矩阵计算加速
 */
TEST(ir_operator, test_sparse_matrix) {
    // Example CSR matrix
    int num_rows = 4;
    int row_ptr[] = {0, 2, 4, 7, 8};
    int col_indices[] = {0, 1, 0, 2, 1, 2, 3, 3};
    double values[] = {10, 20, 30, 40, 50, 60, 70, 80};
    double x[] = {1, 2, 3, 4};
    double y[4] = {0};

    // Performance test
    const int num_trials = 100;
    double total_time = 0.0;
    for (int trial = 0; trial < num_trials; ++trial) {
        auto start = std::chrono::high_resolution_clock::now();

        BatmanInfer::BatmanOperator::csr_matrix_vector_multiply(row_ptr, col_indices, values, x, y, num_rows);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }

    double average_time = total_time / num_trials;
    std::cout << "Average time over " << num_trials << " trials: " << average_time << " seconds" << std::endl;
}

TEST(ir_operator, test_dense_matrix) {
    // Example CSR matrix
    int num_rows = 4;
    int row_ptr[] = {0, 2, 4, 7, 8};
    int col_indices[] = {0, 1, 0, 2, 1, 2, 3, 3};
    double values[] = {10, 20, 30, 40, 50, 60, 70, 80};
    double x[] = {1, 2, 3, 4};
    double y_dense[4] = {0};

    // Convert CSR to dense matrix
    std::vector<std::vector<double>> dense_matrix(num_rows, std::vector<double>(num_rows, 0.0));
    for (int i = 0; i < num_rows; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            dense_matrix[i][col_indices[j]] = values[j];
        }
    }

    // Performance test
    const int num_trials = 100;
    double total_time = 0.0;

    for (int trial = 0; trial < num_trials; ++trial) {
        auto start = std::chrono::high_resolution_clock::now();

        BatmanInfer::BatmanOperator::dense_matrix_vector_multiply(dense_matrix, x, y_dense, num_rows);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }

    double average_time = total_time / num_trials; // Convert to milliseconds
    std::cout << "Average time over " << num_trials << " trials: " << average_time << " seconds" << std::endl;

    // Print result vector for dense matrix
    std::cout << "Result vector y (dense): ";
    for (int i = 0; i < num_rows; ++i) {
        std::cout << y_dense[i] << " ";
    }
    std::cout << std::endl;
}