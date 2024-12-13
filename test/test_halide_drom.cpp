//
// Created by Mason on 2024/12/13.
//

#include <gtest/gtest.h>
#include "Halide.h"

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