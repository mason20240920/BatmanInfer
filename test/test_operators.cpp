//
// Created by Mason on 2024/11/14.
//

#include <gtest/gtest.h>
#include <layer/detail/gemm.hpp>

using namespace BatmanInfer;

TEST(test_operators, gemm_operator) {
    const uint32_t batch_size = 1;
    std::vector<sftensor> inputs(batch_size);
    std::vector<sftensor> outputs(batch_size);

    for (uint32_t i = 0; i < batch_size; ++i) {
        sftensor input = std::make_shared<ftensor>(1, 1, 512);
        input->Rand();

        inputs.at(i) = input;
    }

    std::vector<sftensor> weights;
    std::vector<sftensor> bias;
    sftensor weight_temp = std::make_shared<ftensor>(1, 1000, 512);
    sftensor bias_temp = std::make_shared<ftensor>(1, 1, 512);
    weight_temp->Rand();
    bias_temp->Rand();
    weights.push_back(weight_temp);
    bias.push_back(bias_temp);
    GemmLayer gemmLayer(1, 1, 1, 1000, 512);
    gemmLayer.set_weights(weights);
    gemmLayer.set_bias(bias);
}