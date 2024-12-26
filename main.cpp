#include <iostream>

#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include <arm_neon.h> // ARM Neon SIMD 指令支持

// 定义矩阵类型
using Matrix = std::vector<std::vector<float>>;

// 打印矩阵
void print_matrix(const Matrix &matrix) {
    for (const auto &row : matrix) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// 填充矩阵
Matrix pad_matrix(const Matrix &input, int extra_pad_x, int pad_x, int pad_y) {
    int original_rows = input.size();
    int original_cols = input[0].size();

    // 新的行数和列数
    int new_rows = original_rows + 2 * pad_y;
    int new_cols = original_cols + pad_x + extra_pad_x;

    // 创建填充后的矩阵
    Matrix padded(new_rows, std::vector<float>(new_cols, 0));

    // 复制原始矩阵到填充后的矩阵中
    for (int i = 0; i < original_rows; ++i) {
        for (int j = 0; j < original_cols; ++j) {
            padded[i + pad_y][j + pad_x] = input[i][j];
        }
    }

    return padded;
}

// 使用 ARM Neon 进行矩阵加法示例
void simd_add_matrix(Matrix &matrix, float value) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    // 每次处理 4 个浮点数
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j += 4) {
            // 加载 4 个浮点数
            float32x4_t data = vld1q_f32(&matrix[i][j]);
            // 加上常数值
            float32x4_t result = vaddq_f32(data, vdupq_n_f32(value));
            // 存回矩阵
            vst1q_f32(&matrix[i][j], result);
        }
    }
}

int main(int argc, char ** argv) {

// 原始矩阵 (3x3)
    Matrix matrix = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
    };

    std::cout << "Original Matrix:\n";
    print_matrix(matrix);

    // 填充参数
    int extra_pad_x = 32; // 额外填充 32 列
    int pad_x = 4;        // 基础填充 4 列
    int pad_y = 4;        // 基础填充 4 行

    // 填充矩阵
    Matrix padded_matrix = pad_matrix(matrix, extra_pad_x, pad_x, pad_y);

    std::cout << "Padded Matrix:\n";
    print_matrix(padded_matrix);

    // 使用 SIMD 对填充后的矩阵加上一个常数值
    float add_value = 10.0f;
    simd_add_matrix(padded_matrix, add_value);

    std::cout << "Matrix after SIMD addition:\n";
    print_matrix(padded_matrix);

    return EXIT_SUCCESS;
//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
}
