//
// Created by Mason on 2024/12/8.
//


#include <gtest/gtest.h>
#include <cblas.h>
#include <Halide.h>
#include <chrono> // 用于测量性能

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

using namespace Halide;

// 定义一个包装函数，调用 OpenBLAS 的 cblas_sgemv
extern "C" void openblas_sgemv(halide_buffer_t *A_buf, halide_buffer_t *x_buf, halide_buffer_t *y_buf, int M, int N, float alpha, float beta) {
    // 从 Halide 的 buffer 提取指针
    float *A = (float *)A_buf->host;
    float *x = (float *)x_buf->host;
    float *y = (float *)y_buf->host;

    // 调用 OpenBLAS 的 cblas_sgemv 进行矩阵-向量乘法
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, A, N, x, 1, beta, y, 1);
}

// 定义一个包装函数，直接用 C++ 实现矩阵-向量乘法
extern "C" int custom_sgemv(halide_buffer_t *A_buf,
                             halide_buffer_t *x_buf,
                             halide_buffer_t *y_buf,
                             int M,
                             int N, float alpha, float beta) {

    // 提取指针
    float *A = (float *)A_buf->host;
    float *x = (float *)x_buf->host;
    float *y = (float *)y_buf->host;

    // 矩阵-向量乘法
    for (int i = 0; i < M; i++) {
        float result = 0.0f;
        for (int j = 0; j < N; j++) {
            result += A[i * N + j] * x[j];
        }
        y[i] = alpha * result + beta * y[i];
    }

    return 0;
}

TEST(open_blas_test, demo2) {
    // 定义矩阵和向量的大小
    int M = 2; // 矩阵行数
    int N = 3; // 矩阵列数

    // 定义 Halide 的输入和输出
    Buffer<float> A(N, M); // 输入矩阵
    Buffer<float> x(N);    // 输入向量
    Buffer<float> y(M);    // 输出向量

    // 初始化 A 和 x
    A(0, 0) = 1; A(1, 0) = 2; A(2, 0) = 3;
    A(0, 1) = 4; A(1, 1) = 5; A(2, 1) = 6;
    x(0) = 1; x(1) = 2; x(2) = 3;

    // 定义 Halide 的 Pipeline
    Func matvec;
    Var i;

    // 使用 define_extern 调用 OpenBLAS 的 sgemv
    matvec.define_extern(
            "openblas_sgemv",                     // 外部函数名称
            {A, x, y, Expr(M), Expr(N), 1.0f, 0.0f}, // 参数列表
            Float(32),                            // 输出类型
            1                                    // 输出维度
    );
    // 测量性能
    auto start = std::chrono::high_resolution_clock::now();
    matvec.realize(y);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 输出结果
    for (int i = 0; i < M; i++) {
        std::cout << "y[" << i << "] = " << y(i) << std::endl;
    }

    std::cout << "Execution time without OpenBLAS: " << elapsed.count() << " seconds" << std::endl;
}