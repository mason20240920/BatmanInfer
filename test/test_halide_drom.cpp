//
// Created by Mason on 2024/12/13.
//

#include <gtest/gtest.h>
#include "Halide.h"
#include <data/Tensor.hpp>
#include <cblas.h>

void print_buffer(const Halide::Buffer<int> &buffer) {
    for (int y = 0; y < buffer.height(); y++) {
        for (int x = 0; x < buffer.width(); x++) {
            std::cout << buffer(x, y) << " ";
        }
        std::cout << std::endl;
    }
}

TEST(test_halide_drom, drom_test_1) {
    // 作用域测试
    using namespace Halide;
    Func f;
    Var x;
    RDom r(0, 10);
    f(x) = x; // the initial value
    f(r) = f(r) * 2;
    Buffer<int> result = f.realize({10});

    print_buffer(result);
}

TEST(test_halide_rdom, rdom_test_2) {
    using namespace Halide;
    Func f;
    Var x;
    RDom r(2, 18);
    f(x) = 1;
    f(r) = f(r - 1) + f(r - 2);
    Buffer<int> result = f.realize({10});

    print_buffer(result);
}

TEST(test_halide_rdom, rdom_multi_dim) {
    using namespace Halide;

    // Define the input buffer
    Buffer<int> input(20, 20);

    // Fill the input buffer with some example data
    for (int y = 0; y < 20; y++) {
        for (int x = 0; x < 20; x++) {
            input(x, y) = x + y;
        }
    }

    // Define a Halide function
    Func sum;
    Var x, y;

    // Define a reduction domain over the region (0, 0) to (20, 20)
    RDom r(0, 20, 0, 20);

    sum() = 0;
    sum() += input(r.x, r.y);

    Buffer<int> result = sum.realize();

    // Print the result
    std::cout << "The sum of the values in the buffer is: " << result(0) << std::endl;
}

void transpose_buffer(halide_buffer_t input) {
    using namespace Halide;
    // 获取原始维度信息
    halide_dimension_t* dim = input.dim;

    // 交换 stride 和 extent
    int32_t temp_extent = dim[0].extent;
    int32_t temp_stride = dim[0].stride;

    dim[0].extent = dim[1].extent;
    dim[0].stride = dim[1].stride;

    dim[1].extent = temp_extent;
    dim[1].stride = temp_stride;
}

TEST(test_halide_rdom, transpose_matrix) {
    using namespace BatmanInfer;
    // Example usage
    // Create an example halide_buffer_t (e.g., 4x3 image)
    sftensor input_tensor = std::make_shared<ftensor>(3, 4);
    std::vector<float> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    input_tensor->Fill(values, true);
    input_tensor->Show();
    sftensor output_tensor = std::make_shared<ftensor>(4, 3);
    // Perform transpose
    transpose_buffer(input_tensor->data());

    input_tensor->Show();
}

/**
 * @brief 提取 halide_buffer_t 的数据指针
 * @param buffer
 * @return
 */
float* extract_data_from_halide_buffer(const halide_buffer_t* buffer) {
    if (buffer->host == nullptr)
        std::cerr << "halide_buffer_t has no host data.";
    return (float*)buffer->host;
}