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
    // 调用 InitIm2ColWeight 函数ju
    conv_layer->InitIm2ColWeight();
}

TEST(test_registry, create_layer_conv_forward) {
    std::map<std::string, sftensor> inputs{{"input", std::make_shared<ftensor>(4, 4, 4)}};
    std::map<std::string, sftensor> outputs{{"output", nullptr}};

    const uint32_t kernel_h = 3;
    const uint32_t kernel_w = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t kernel_count = 2;
    sftensor weights = std::make_shared<ftensor>(kernel_count,
                                                 2,
                                                 kernel_h,
                                                 kernel_w);
    inputs.at("input")->Fill(std::vector<float>{1, 2, 3, 4,
                                                5, 6, 7, 8,
                                                9, 10, 11, 12,
                                                13, 14, 15, 16,
                                                17, 18, 19, 20,
                                                21, 22, 23, 24,
                                                25, 26, 27, 28,
                                                29, 30, 31, 32,
                                                33, 34, 35, 36,
                                                37, 38, 39, 40,
                                                41, 42, 43, 44,
                                                45, 46, 47, 48,
                                                49, 50, 51, 52,
                                                53, 54, 55, 56,
                                                57, 58, 59, 60,
                                                61, 62, 63, 64});

    ConvolutionLayer conv_layer(kernel_count,
                                4,
                                kernel_h,
                                kernel_w,
                                0, 0, 0, 0,
                                stride_h, stride_w,
                                2,
                                false);
    conv_layer.set_weights(weights);
    conv_layer.Forward(inputs, outputs);
//    outputs.at(0)->Show();
}