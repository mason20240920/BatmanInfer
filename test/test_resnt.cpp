//
// Created by Mason on 2024/11/12.
//

#include <gtest/gtest.h>
#include <vector>
#include <runtime/runtime_ir.hpp>

TEST(test_network, resnet1) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/simple_conv_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build("Input", "Output");
    ASSERT_EQ(int(graph.graph_state()), 0);

    // Flat list of values obtained from PyTorch
    std::vector<float> values = {
            0.5400, 0.0000, 0.0800, 0.5300,
            1.0000, 0.6200, 0.4700, 0.9200
    };


    std::shared_ptr<Tensor<float>> my_tensor = std::make_shared<Tensor<float>>(2, 2, 2);
    my_tensor->Fill(values);
    my_tensor->Show();
    std::vector<sftensor> input{my_tensor};
//    input.at(0)->Show();

    auto outputs = graph.Forward(input, true);
    std::cout << outputs.size() << std::endl;
    outputs.at(0)->Show();
//    std::cout << "Hello World" << std::endl;
}

// 测试拓扑结构是否正常
TEST(test_network, resnet2) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/resnet18.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build("Input", "Output");
    ASSERT_EQ(int(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor = std::make_shared<ftensor>(3, 2, 2);
    input_tensor->Ones();
    std::vector<sftensor> input{input_tensor};
    input.at(0)->Show();

    auto outputs = graph.Forward(input, true);
    outputs.at(0)->Show();
}