//
// Created by Mason on 2024/11/12.
//

#include <gtest/gtest.h>
#include <layer/detail/convolution.hpp>

using namespace BatmanInfer;

class ConvolutionLayerTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化卷积层参数
        // 卷积核数量
        output_channel = 4;
        // 输入的通道数
        in_channel = 4;
        // 卷积核高度
        kernel_h = 3;
        // 卷积核宽度
        kernel_w = 3;
        // padding的高度
        padding_h = 1;
        // padding的宽度
        padding_w = 1;
        // 纵向步长
        stride_h = 1;
        // 横向步长
        stride_w = 1;
        // 组的数量
        groups = 2;
        use_bias = false;

        // 创建 ConvolutionLayer 实例
        conv_layer = std::make_shared<ConvolutionLayer>(
                output_channel,
                in_channel,
                kernel_h,
                kernel_w,
                padding_h,
                padding_h,
                padding_w,
                padding_w,
                stride_h,
                stride_w,
                groups,
                use_bias
        );
    }

    uint32_t output_channel{}, in_channel{}, kernel_h{}, kernel_w{};
    uint32_t padding_h{}, padding_w{}, stride_h{}, stride_w{}, groups{};
    bool use_bias{};
    std::shared_ptr<ConvolutionLayer> conv_layer;
};



TEST_F(ConvolutionLayerTest, TestInitIm2ColWeight) {
    // 调用 InitIm2ColWeight 函数
    conv_layer->InitIm2ColWeight();
}

TEST(test_registry, create_layer_conv_forward) {
    const uint32_t batch_size = 1;
    std::vector<sftensor> inputs(batch_size);
    std::vector<sftensor> outputs(batch_size);

    const uint32_t in_channel = 2;
    for (uint32_t i = 0; i < batch_size; ++i) {
        sftensor input = std::make_shared<ftensor>(in_channel, 4, 4);
        input->data().slice(0) = "1,2,3,4;"
                                 "5,6,7,8;"
                                 "9,10,11,12;"
                                 "13,14,15,16;";

        input->data().slice(1) = "1,2,3,4;"
                                 "5,6,7,8;"
                                 "9,10,11,12;"
                                 "13,14,15,16;";

        inputs.at(i) = input;
    }

    const uint32_t kernel_h = 3;
    const uint32_t kernel_w = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t kernel_count = 2;
    std::vector<sftensor> weights;
    for (uint32_t i = 0; i < kernel_count; ++i) {
        sftensor kernel = std::make_shared<ftensor>(in_channel,
                                                    kernel_h,
                                                    kernel_w);
        kernel->data().slice(0) = arma::fmat("1,2,3;"
                                              "3,2,1;"
                                              "1,2,3;");
        kernel->data().slice(1) = arma::fmat("1,2,3;"
                                             "3,2,1;"
                                             "1,2,3;");
        weights.push_back(kernel);
    }
    ConvolutionLayer conv_layer(kernel_count,
                                in_channel,
                                kernel_h,
                                kernel_w,
                                0, 0, 0, 0,
                                stride_h, stride_w,
                                1,
                                false);
    conv_layer.set_weights(weights);
    conv_layer.Forward(inputs, outputs);
    outputs.at(0)->Show();
}