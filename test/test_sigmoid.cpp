//
// Created by Mason on 2024/11/1.
//
#include <gtest/gtest.h>
#include <layer/abstract/layer_factory.hpp>
#include <Halide.h>
#include <omp.h>
#include <random>

using namespace BatmanInfer;
using namespace Halide;

TEST(test_registry, create_layer_find) {
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "nn.Sigmoid";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    // 评价注册是否成功
    ASSERT_NE(layer, nullptr);
}

void compute_softmax(halide_buffer_t &input_buffer,
                     halide_buffer_t &output_buffer,
                     const int axis) {
    // 将输入和输出包装为 Halide::Buffer
    Buffer<float> input(input_buffer);
    Buffer<float> output(output_buffer);

    // 检查输入和输出缓冲区是否定义
    if (!input.defined() || !output.defined()) {
        std::cerr << "Buffer not properly defined" << std::endl;
        return;
    }

    // 定义 Halide 变量
    Var x("x"), y("y"), z("z");

    Func max_val("max_val"), exp_values("exp_values"), sum_exp("sum_exp"), softmax("softmax");

    if (input.dimensions() == 1) {
        // Handle 1D case
        RDom r(0, input.width(), "r");
        max_val() = maximum(input(r));
        exp_values(x) = exp(input(x) - max_val());
        sum_exp() = sum(exp_values(r));
        softmax(x) = exp_values(x) / sum_exp();
        softmax.parallel(x).vectorize(x, 8);
    } else if (input.dimensions() == 2) {
        // Handle 2D case with axis
        if (axis == 0) { // Column-wise
            RDom r(0, input.height(), "r");
            max_val(x) = maximum(input(x, r));
            exp_values(x, y) = exp(input(x, y) - max_val(x));
            sum_exp(x) = sum(exp_values(x, r));
            softmax(x, y) = exp_values(x, y) / sum_exp(x);
            softmax.parallel(x).vectorize(y, 8);
        } else {
            RDom r(0, input.width(), "r");
            // 注意: 这里y是固定的,
            max_val(y) = maximum(input(r, y));
            exp_values(x, y) = exp(input(x, y) - max_val(y));
            sum_exp(y) = sum(exp_values(r, y));
            softmax(x, y) = exp_values(x, y) / sum_exp(y);
            // OpenMP and SIMD
            softmax.parallel(y).vectorize(x, 8);
        }
    } else if (input.dimensions() == 3) {
        // Handle 3D case with axis
        if (axis == 0) { // Depth-wise (z-axis)
            RDom r(0,
                   input.dim(2).extent(),
                   "r");
            max_val(x, y) = maximum(input(x, y, r));
            exp_values(x, y, z) = exp(input(x, y, z) - max_val(x, y));
            sum_exp(x, y) = sum(exp_values(x, y, r));
            softmax(x, y, z) = exp_values(x, y, z) / sum_exp(x, y);
        } else if (axis == 1) {
            RDom r(0, input.dim(1).extent(), "r");
            max_val(x, z) = maximum(input(x, r, z));
            exp_values(x, y, z) = exp(input(x, y, z) - max_val(x, z));
            sum_exp(x, z) = sum(exp_values(x, r, z));
            softmax(x, y, z) = exp_values(x, y, z) / sum_exp(x, z);
        } else {
            RDom r(0, input.dim(0).extent(), "r");
            max_val(y, z) = maximum(input(r, y, z));
            exp_values(x, y, z) = exp(input(x, y, z) - max_val(y, z));
            sum_exp(y, z) = sum(exp_values(r, y, z));
            softmax(x, y, z) = exp_values(x, y, z) / sum_exp(y, z);
        }
    }

    softmax.realize(output);
}

void compute_max_val(halide_buffer_t &input_halide,
                     halide_buffer_t &output_halide,
                     int axis) {
    Buffer<float> input(input_halide);
    Buffer<float> output(output_halide);
    // 定义变量
    Var x("x"), y("y"), z("z");

    // 定义函数
    Func max_val("max_val"), exp_val("exp_val"), sum_exp("sum_exp"), softmax("softmax");

    if (axis == 0) {
        // 定义一个归约域，范围是 z 轴的所有值
        RDom r(0, input.dim(2).extent(), "r");

        // 对 z 轴进行归约，找到最大值
        max_val(x, y) = maximum(input(x, y, r));

        // 计算exp(input - max_val)
        exp_val(x, y, z) = exp(input(x, y, z) - max_val(x, y));

        sum_exp(x, y) = sum(exp_val(x, y, r));

        // 计算softmax
        softmax(x, y, z) = exp_val(x, y, z) / sum_exp(x, y);

        // 创建一个 Buffer 来存储结果
//        Buffer<float> result(output.width(), output.height(), output.channels());

        // 实现并填充结果 Buffer
        softmax.realize(output);

//        result.copy_to_host();
        // 打印结果（可选）
        // 打印结果为矩阵形式
        for (int k = 0; k < output.channels(); k++) {
            printf("Channel %d:\n", k);
            for (int i = 0; i < output.width(); i++) {
                for (int j = 0; j < output.height(); j++) {
                    printf("%f ", output(i, j, k));
                }
                printf("\n");
            }
            printf("\n"); // 分隔不同的通道
        }
    } else if (axis == 1) {
        // 定义一个归约域，范围是 y 轴的所有值
        RDom r(0, input.dim(1).extent(), "r");

        // 对 y 轴进行归约，找到最大值
        max_val(x, z) = maximum(input(x, r, z));

        // 计算exp(input - max_val)
        exp_val(x, y, z) = exp(input(x, y, z) - max_val(x, z));

        // 对 y 轴进行求和
        sum_exp(x, z) = sum(exp_val(x, r, z));

        // 计算softmax
        softmax(x, y, z) = exp_val(x, y, z) / sum_exp(x, z);

        // 创建一个 Buffer 来存储结果
        Buffer<float> result(output);

        // 实现并填充结果 Buffer
        softmax.realize(result);

        // 打印结果为矩阵形式
//        for (int z = 0; z < result.channels(); z++) {
//            printf("Channel %d:\n", z);
//            for (int x = 0; x < result.width(); x++) {
//                for (int y = 0; y < result.height(); y++) {
//                    printf("%f ", result(x, y, z));
//                }
//                printf("\n");
//            }
//            printf("\n"); // 分隔不同的通道
//        }
    } else if (axis == 2)  {
        // 定义一个归约域，范围是 x 轴的所有值
        RDom r(0, input.dim(0).extent(), "r");

        // 对 y 轴进行归约，找到最大值
        max_val(y, z) = maximum(input(r, y, z));

        // 计算exp(input - max_val)
        exp_val(x, y, z) = exp(input(x, y, z) - max_val(y, z));

        // 对 x 轴进行求和
        sum_exp(y, z) = sum(exp_val(r, y, z));

        // 计算softmax
        softmax(x, y, z) = exp_val(x, y, z) / sum_exp(y, z);

        // 创建一个 Buffer 来存储结果
        Buffer<float> result(input.width(), input.height(), input.channels());

        // 实现并填充结果 Buffer
        softmax.realize(result);

        // 打印结果为矩阵形式
        for (int z = 0; z < result.channels(); z++) {
            printf("Channel %d:\n", z);
            for (int x = 0; x < result.height(); x++) {
                for (int y = 0; y < result.width(); y++) {
                    printf("%f ", result(y, x, z));
                }
                printf("\n");
            }
            printf("\n"); // 分隔不同的通道
        }
    }
    // Add similar blocks for axis == 1 and axis == 2 if needed
}

TEST(test_halide_softmax, softmax_tensor_1_dim) {
    // 示例数据
    const int size = 768;
    float data[size];
    float result[size] = {0};
    // 示例初始化：可以用循环或其他方法填充 data
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i + 1); // 或者其他初始化方式
    }

    // 创建 Halide 缓冲区
    halide_dimension_t shape[1] = {{0, size, 1}};
    halide_buffer_t input_buffer = {0};
    input_buffer.host = (uint8_t *) data;
    input_buffer.dimensions = 1;
    input_buffer.dim = shape;
    input_buffer.type = halide_type_of<float>();

    halide_buffer_t output_buffer = {0};
    output_buffer.host = (uint8_t *) result;
    output_buffer.dimensions = 1;
    output_buffer.dim = shape;
    output_buffer.type = halide_type_of<float>();

    // 计算 Softmax
    compute_softmax(input_buffer, output_buffer, 0);
    // 初始化输入和输出缓冲区

    int num_runs = 10;
    double total_time = 0.0;

    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        compute_softmax(input_buffer, output_buffer, 0);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }

    double average_time = total_time / num_runs;
    std::cout << "Average execution time over " << num_runs << " runs: "
              << average_time << " seconds." << std::endl;
}

TEST(test_halide_softmax, softmax_tensor_1_or_2_dim) {
    const int width = 768;
    const int height = 8; // For 2D tensor

    float data[height][width];
    float result[height][width] = {0};

    for (auto &y: data) {
        for (int x = 0; x < width; ++x) {
            y[x] = static_cast<float>(x + 1); // 或者其他初始化方式
        }
    }

    // 创建 Halide 缓冲区
    halide_dimension_t shape_2d[2] = {{0, width,  1},
                                      {0, height, width}};
    halide_buffer_t input_buffer = {0};
    input_buffer.host = (uint8_t *) data;
    input_buffer.dimensions = 2; // For 2D tensor
    input_buffer.dim = shape_2d;
    input_buffer.type = halide_type_of<float>();

    halide_buffer_t output_buffer = {0};
    output_buffer.host = (uint8_t *) result;
    output_buffer.dimensions = 2; // For 2D tensor
    output_buffer.dim = shape_2d;
    output_buffer.type = halide_type_of<float>();

    // 初始化输入和输出缓冲区

    int num_runs = 10;
    double total_time = 0.0;

    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        compute_softmax(input_buffer, output_buffer, 1);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }

    double average_time = total_time / num_runs;
    std::cout << "Average execution time over " << num_runs << " runs: "
              << average_time << " seconds." << std::endl;
}

TEST(test_halide_softmax, softmax_tensor_3_dim) {
    const int depth = 2;
    const int height = 4;
    const int width = 4;

    float data[depth][height][width];
    float result[depth][height][width] = {0};

    // 初始化数据
    for (auto &z: data) {
        for (auto &y: z) {
            for (int x = 0; x < width; ++x) {
                y[x] = static_cast<float>(x + 1); // 或者其他初始化方式
            }
        }
    }

    for (int z = 0; z < depth; ++z) {
        std::cout << "Depth " << z << ":\n";
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::cout << data[z][y][x] << " ";
            }
            std::cout << "\n"; // 换行，表示一行结束
        }
        std::cout << "\n"; // 换行，分隔不同的深度
    }

    // 创建 Halide 缓冲区
    halide_dimension_t shape_3d[3] = {{0, width,  1},
                                      {0, height, width},
                                      {0, depth,  width * height}};
    halide_buffer_t input_buffer = {0};
    input_buffer.host = (uint8_t *) data;
    input_buffer.dimensions = 3; // For 3D tensor
    input_buffer.dim = shape_3d;
    input_buffer.type = halide_type_of<float>();

    halide_buffer_t output_buffer = {0};
    output_buffer.host = (uint8_t *) result;
    output_buffer.dimensions = 3; // For 3D tensor
    output_buffer.dim = shape_3d;
    output_buffer.type = halide_type_of<float>();

    // 调用 compute_softmax，假设沿着 width (x-axis) 进行 softmax
//    compute_max_val(input_buffer, output_buffer, 0);
    compute_softmax(input_buffer, output_buffer, 2);

    // 打印结果为矩阵形式
    for (int z = 0; z < depth; ++z) {
        std::cout << "Depth " << z << ":\n";
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::cout << result[z][y][x] << " ";
            }
            std::cout << "\n"; // 换行，表示一行结束
        }
        std::cout << "\n"; // 换行，分隔不同的深度
    }

}

TEST(test_sigmoids, test_sigmoid_operator) {
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "Sigmoid";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);

    sftensor input_tensor = std::make_shared<ftensor>(1, 2, 8, 8);
    input_tensor->Ones();
    input_tensor->Show();
    std::map<std::string, sftensor> input_map {
            {"input", input_tensor}
    };

    sftensor output = std::make_shared<ftensor>(1, 2, 8, 8);
    std::map<std::string, sftensor> output_map {
            {"output", output}
    };

    layer->Forward(input_map, output_map);

    output->Show();
}