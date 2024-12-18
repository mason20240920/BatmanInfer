//
// Created by Mason on 2024/11/14.
//

#include <gtest/gtest.h>
#include <layer/detail/gemm.hpp>
#include <data/tensor_util.hpp>
#include <layer/detail/concat.hpp>
#include <layer/detail/expand.hpp>
#include <layer/detail/trilu.hpp>
#include <vector>
#include <runtime/runtime_ir.hpp>

using namespace BatmanInfer;

TEST(test_operators, gemm_operator) {
    const uint32_t batch_size = 1;
    std::vector<sftensor> inputs(batch_size);
    std::vector<sftensor> outputs(batch_size);

    for (uint32_t i = 0; i < batch_size; ++i) {
        sftensor input = std::make_shared<ftensor>(1, 1, 5);
        input->Rand();
        input->Show();

        inputs.at(i) = input;
    }

    std::vector<sftensor> weights;
    std::vector<sftensor> bias;
    sftensor weight_temp = std::make_shared<ftensor>(1, 5, 3);
    sftensor bias_temp = std::make_shared<ftensor>(1, 3, 1);
    weight_temp->Rand();
    weight_temp->Transpose();
    bias_temp->Rand();
    weight_temp->Show();
    bias_temp->Show();
    weights.push_back(weight_temp);
    auto shapes = bias_temp->shapes();
    bias.push_back(bias_temp);
    GemmLayer gemmLayer(1, 1, 1, 3, 5);
    gemmLayer.set_weights(weights);
    gemmLayer.set_bias(bias);
    gemmLayer.Forward(inputs, outputs);
    outputs.at(0)->Show();
}

TEST(test_operators, tensor_matrix_mul) {
    auto tensor1 = std::make_shared<Tensor<float>>(3, 1, 2);
    tensor1->Rand();
    tensor1->Show();
    auto tensor2 = std::make_shared<Tensor<float>>(3, 2, 1);
    tensor2->Rand();
    tensor2->Show();
    auto tensor3 = MatrixMultiply(tensor1, tensor2);
    tensor3->Show();
}

TEST(test_operators, tensor_concat) {
    auto tensor1 = std::make_shared<Tensor<float>>(2, 2, 2);
    tensor1->Rand();
    tensor1->Show();
    auto tensor2 = std::make_shared<Tensor<float>>(2, 2, 2);
    tensor2->Rand();
    tensor2->Show();
    std::vector<sftensor> inputs{tensor1, tensor2};
    std::vector<sftensor> outputs(2);
    ConcatLayer concatLayer(0);
    concatLayer.Forward(inputs, outputs);
    outputs.at(0)->Show();
    CHECK(outputs.at(0)->shapes()[0] == 2);
    CHECK(outputs.at(0)->shapes()[1] == 2);
    CHECK(outputs.at(0)->shapes()[2] == 2);
}

TEST(test_operators, tensor_concat2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/concat_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build({ "input1", "input2" }, { "output" });
    ASSERT_EQ(int(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor1 = std::make_shared<ftensor>(2, 224, 224);
    input_tensor1->Ones();
    std::shared_ptr<ftensor> input_tensor2 = std::make_shared<ftensor>(1, 224, 224);
    input_tensor2->Ones();
    std::vector<sftensor> input1{input_tensor1};
    std::vector<sftensor> input2{input_tensor2};

    auto outputs = graph.Forward({ input1, input2 }, true);
}

TEST(test_operators, tensor_expand1) {
    using namespace BatmanInfer;
    auto tensor = std::make_shared<Tensor<float>>(1, 2, 2);
    tensor->Ones();
    tensor->Show();
    auto output_tensor = std::make_shared<ftensor>(2, 2, 2);
    std::vector<sftensor> inputs{tensor};
    std::vector<sftensor> outputs{output_tensor};
    ExpandLayer expandLayer{};
    expandLayer.Forward(inputs, outputs);
    outputs.at(0)->Show();
}

TEST(test_operators, tensor_expand2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/expands_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build({ "input" }, { "output" });
    ASSERT_EQ(int(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor1 = std::make_shared<ftensor>(1, 1, 1);
    input_tensor1->Ones();
    std::vector<sftensor> input1{input_tensor1};

    auto outputs = graph.Forward({ input1 }, true);
    CHECK(outputs.size() == 1);
}

TEST(test_operators, tensor_trilu) {
    using namespace BatmanInfer;
    auto tensor = std::make_shared<Tensor<float>>(3, 3, 3);
    tensor->Ones();
    tensor->Show();
    auto output_tensor = std::make_shared<ftensor>(3, 3, 3);
    std::vector<sftensor> inputs{tensor};
    std::vector<sftensor> outputs{output_tensor};
    TriluLayer triluLayer{1};
    triluLayer.Forward(inputs, outputs);
    outputs.at(0)->Show();
}

TEST(test_operators, tensor_trilu2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/trilu_operator.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build({ "input" }, { "output" });
    ASSERT_EQ(int(graph.graph_state()), 0);

    auto tensor = std::make_shared<Tensor<float>>(3, 3, 3);
    tensor->Ones();
    tensor->Show();
    std::vector<sftensor> inputs{tensor};

    auto outputs = graph.Forward({inputs}, true);
}

TEST(test_operators, tensor_equal2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/equal_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build({ "input" }, { "output" });
    ASSERT_EQ(int(graph.graph_state()), 0);

    auto tensor = std::make_shared<Tensor<float>>(1, 2, 2);
    tensor->Fill(std::vector<float>{10, 10, 1, 1});
    tensor->Show();
    std::vector<sftensor> inputs{tensor};

    auto outputs = graph.Forward({inputs}, true);
}

TEST(test_operators, tensor_where2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/where_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build({ "condition" }, { "output" });
    ASSERT_EQ(int(graph.graph_state()), 0);

    auto tensor = std::make_shared<Tensor<float>>(1, 4);
    tensor->Fill(std::vector<float>{10, 10, 1, 1});
    tensor->Show();
    std::vector<sftensor> inputs{tensor};

    auto outputs = graph.Forward({inputs}, true);
}

TEST(test_operators, tensor_cast2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/cast_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build({ "input" }, { "output" });
    ASSERT_EQ(int(graph.graph_state()), 0);

    auto tensor = std::make_shared<Tensor<float>>(2, 3, 3);
    tensor->Fill(1);
    tensor->Show();
    std::vector<sftensor> inputs{tensor};

    auto outputs = graph.Forward({inputs}, true);
}

TEST(test_operators, tensor_sqrt2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/sqrt_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -1);
    graph.Build({ "input" }, { "output" });
    ASSERT_EQ(static_cast<int>(graph.graph_state()), 0);

    const auto tensor = std::make_shared<Tensor<float>>(2, 3, 3);
    tensor->Fill(9);
    tensor->Show();
    std::vector<sftensor> inputs{tensor};

    const auto outputs = graph.Forward({inputs}, true);
}

TEST(test_operators, tensor_div2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/div_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -1);
    graph.Build({ "A" }, { "C" });
    ASSERT_EQ(static_cast<int>(graph.graph_state()), 0);

    const auto tensor = std::make_shared<Tensor<float>>(2, 2, 2);
    tensor->Fill(9);
    tensor->Show();
    std::vector<sftensor> inputs{tensor};

    const auto outputs = graph.Forward({inputs}, true);
}

TEST(test_operators, tensor_matmul2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/matmul_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -1);
    graph.Build({ "input_x", "input_y" }, { "output" });
    ASSERT_EQ(static_cast<int>(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor1 = std::make_shared<ftensor>(2, 3, 4);
    input_tensor1->Ones();
    std::shared_ptr<ftensor> input_tensor2 = std::make_shared<ftensor>(2, 4, 5);
    input_tensor2->Ones();
    std::vector<sftensor> input1{input_tensor1};
    std::vector<sftensor> input2{input_tensor2};

    const auto outputs = graph.Forward({input1, input2}, true);
}

TEST(test_operators, tesnor_mul2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/elementwise_mul.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -1);
    graph.Build({ "input_A", "input_B" }, { "output_C" });
    ASSERT_EQ(static_cast<int>(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor1 = std::make_shared<ftensor>(3, 2, 2);
    input_tensor1->Fill(2);
    std::shared_ptr<ftensor> input_tensor2 = std::make_shared<ftensor>(3, 2, 2);
    input_tensor2->Fill(3);
    std::vector<sftensor> input1{input_tensor1};
    std::vector<sftensor> input2{input_tensor2};

    const auto outputs = graph.Forward({input1, input2}, true);
}

TEST(test_operators, tensor_transpose2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/transpose.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -1);
    graph.Build({ "input" }, { "output" });
    ASSERT_EQ(static_cast<int>(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor = std::make_shared<ftensor>(2, 3, 4);
    std::vector<float> vec(24);
    std::iota(vec.begin(), vec.end(), 1.0f);
    input_tensor->Fill(vec, true);
    input_tensor->Show();
    std::vector<sftensor> input{input_tensor};

    const auto outputs = graph.Forward({input}, true);
}

TEST(test_operators, tensor_reshape2) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/attention_reshape_static.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -1);
    graph.Build({ "hidden_states" }, { "reshaped" });
    ASSERT_EQ(static_cast<int>(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor = std::make_shared<ftensor>(1, 2, 768);
    input_tensor->Ones();
    std::vector<sftensor> input{input_tensor};

    const auto outputs = graph.Forward({input}, true);
    auto ret = outputs.at("reshaped");
    ret.at(0).at(0)->Show();
}

TEST(test_operators, tensor_split3) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/operators/split_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -1);
    graph.Build({ "input" }, { "output_2", "output_1", "output_0" });
    ASSERT_EQ(static_cast<int>(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor = std::make_shared<ftensor>(1, 2, 2304);
    input_tensor->Ones();
    std::vector<sftensor> input{input_tensor};
    const auto outputs = graph.Forward({input}, true);
    outputs.at("output_2").at(0).at(0)->Show();
    outputs.at("output_1").at(0).at(0)->Show();
}

TEST(test_operators, tensor_fuse1) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/fuse_operators/simplified_custom_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -1);
    graph.Build({ "hidden_states" }, { "output3", "output2", "output1" });
    ASSERT_EQ(static_cast<int>(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor = std::make_shared<ftensor>(1, 2, 3);
    input_tensor->Ones();
    input_tensor->Show();
    std::vector<sftensor> input{input_tensor};
    const auto outputs = graph.Forward({input}, true);
    for (auto [_, output] : outputs) {
        output.at(0).at(0)->Show();
    }
}


TEST(test_operators, tensor_self_attention) {
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/simplified_gpt2_attention.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(static_cast<int>(graph.graph_state()), -1);
    graph.Build({ "hidden_states" }, { "attention_output" });
    ASSERT_EQ(static_cast<int>(graph.graph_state()), 0);
}