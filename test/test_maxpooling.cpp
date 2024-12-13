//
// Created by Mason on 2024/11/6.
//

#include <layer/abstract/layer_factory.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <Halide.h>

using namespace BatmanInfer;

TEST(test_registry, create_layer_poolingforward) {
    using namespace Halide;
    // 初始化输入数据
    int input_w = 2;
    int input_h = 2;
    int input_c = 1; // 假设只有一个通道

    Buffer<float> input(input_w, input_h, input_c);

    // 填充输入数据
    for (int c = 0; c < input_c; ++c) {
        for (int y = 0; y < input_h; ++y) {
            for (int x = 0; x < input_w; ++x) {
                input(x, y, c) = static_cast<float>(x + y); // 简单地将值设为 x + y
            }
        }
    }

    // 定义池化和步幅参数
    int pooling_h = 2;
    int pooling_w = 2;
    int stride_h = 2;
    int stride_w = 2;
    int pad_h = 2; // 高度方向上下各填充1
    int pad_w = 2; // 宽度方向左右各填充1

    // 计算输出尺寸（考虑填充）
    int output_h = (input_h + 2 * pad_h - pooling_h) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - pooling_w) / stride_w + 1;

    Buffer<float> output(output_w, output_h, input_c);

    // 定义 Halide 变量
    Var x("x"), y("y"), c("c");

    // 定义输入的填充函数
    Func padded_input("padded_input");
    padded_input(x, y, c) = select(x < 0 || x >= input_w || y < 0 || y >= input_h,
                                   0.0f,
                                   input(clamp(x, 0, input_w-1), clamp(y, 0, input_h-1), c));

    // 定义池化函数
    RDom r(0, pooling_w, 0, pooling_h);
    Func pool("pool");

    // 修改池化操作
    pool(x, y, c) = maximum(
            padded_input(x * stride_w + r.x - pad_w,
                         y * stride_h + r.y - pad_h,
                         c)
    );


    // 计算输出
    pool.realize(output);

    // 输出结果
    std::cout << "Input:\n";
    for (int y = 0; y < input_h; ++y) {
        for (int x = 0; x < input_w; ++x) {
            std::cout << input(x, y, 0) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nOutput:\n";
    for (int y = 0; y < output_h; ++y) {
        for (int x = 0; x < output_w; ++x) {
            std::cout << output(x, y, 0) << " ";
        }
        std::cout << std::endl;
    }
}

TEST(test_registry, create_layer_poolingforward_1) {
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "MaxPool";
    std::vector<int> strides{2, 2};
    std::shared_ptr<RuntimeParameter> stride_param = std::make_shared<RuntimeParameterIntArray>(strides);
    op->params.insert({"strides", stride_param});

    std::vector<int> kernel{2, 2};
    std::shared_ptr<RuntimeParameter> kernel_param = std::make_shared<RuntimeParameterIntArray>(kernel);
    op->params.insert({"kernel_shape", kernel_param});

    std::vector<int> paddings{1, 1, 1, 1};
    std::shared_ptr<RuntimeParameter> padding_param = std::make_shared<RuntimeParameterIntArray>(paddings);
    op->params.insert({"pads", padding_param});

    std::shared_ptr<Layer> layer;
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);

    sftensor tensor = std::make_shared<ftensor>(1, 1, 4, 4);
    tensor->Ones();
    tensor->Show();
    std::map<std::string, sftensor> inputs{{"input", tensor}};
    std::map<std::string, sftensor> outputs{{"output", nullptr}};
    layer->Forward(inputs, outputs);
//    tensor->data().slice(0) = input;
//    std::vector<sftensor> inputs(1);
//    inputs.at(0) = tensor;
//    std::vector<sftensor> outputs(1);
//    layer->Forward(inputs, outputs);
//
//    ASSERT_EQ(outputs.size(), 1);
    outputs.at("output")->Show();
}