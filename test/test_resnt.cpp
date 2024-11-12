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
    graph.Build("input", "output");
    ASSERT_EQ(int(graph.graph_state()), 0);

    std::vector<sftensor> input{std::make_shared<Tensor<float>>(1, 1, 1)};
    auto outputs = graph.Forward(input, true);
}