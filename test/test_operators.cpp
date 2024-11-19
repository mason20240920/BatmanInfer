//
// Created by Mason on 2024/11/14.
//

#include <gtest/gtest.h>
#include <layer/detail/gemm.hpp>
#include <data/tensor_util.hpp>
#include <vector>

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