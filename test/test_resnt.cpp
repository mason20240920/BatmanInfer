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
            0.2599,   0.5737, -1.6361,  0.4757
    };

    std::shared_ptr<Tensor<float>> my_tensor = std::make_shared<Tensor<float>>(1, 2, 2);
    my_tensor->Fill(values);
    std::vector<sftensor> input{my_tensor};
    input.at(0)->Show();

    auto outputs = graph.Forward(input, true);
    outputs.at(0)->Show();
    std::cout << "Hello World" << std::endl;
}