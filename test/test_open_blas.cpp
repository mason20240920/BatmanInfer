//
// Created by Mason on 2024/12/8.
//


#include <gtest/gtest.h>
#include <cblas.h>

TEST(open_blas_test, demo1) {
    // 定义矩阵 A 和向量 x, y
    float A[6] = {1, 2, 3, 4, 5, 6}; // 2x3 矩阵
    float x[3] = {1, 2, 3};          // 3x1 向量
    float y[2] = {0, 0};             // 2x1 向量（结果）

    // 矩阵-向量乘法：y = alpha * A * x + beta * y
    float alpha = 1.0f, beta = 0.0f;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 3, alpha, A, 3, x, 1, beta, y, 1);
    // 输出结果
    std::cout << "y = [" << y[0] << ", " << y[1] << "]" << std::endl;

}