//
// Created by Mason on 2024/11/12.
//

#include <gtest/gtest.h>
#include <vector>
#include <runtime/runtime_ir.hpp>
#include <opencv2/opencv.hpp>
#include "layer/detail/softmax.hpp"

using namespace BatmanInfer;

sftensor PreProcessImage(const cv::Mat &image) {
    // 确保输入图像不为空
    assert(!image.empty());

    // 调整图像大小到 224x224
    cv::Mat resize_image;
    cv::resize(image, resize_image, cv::Size(224, 224));

    // 将图像从 BGR 转换为 RGB
    cv::Mat rgb_image;
    cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

    // 将图像数据类型转换为 32 位浮点型，每个像素有 3 个通道
    rgb_image.convertTo(rgb_image, CV_32FC3);

    // 分割 RGB 图像为单独的通道
    std::vector<cv::Mat> split_images;
    cv::split(rgb_image, split_images);

    // 定义输入图像的宽度、高度和通道数
    uint32_t input_w = 224;
    uint32_t input_h = 224;
    uint32_t input_c = 3;

    // 创建一个共享指针，指向一个新的 Tensor<float> 对象
    sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

    uint32_t index = 0;
    for (const auto &split_image : split_images) {
        // 确保每个分割图像的像素总数与输入图像的大小一致
        assert(split_image.total() == input_w * input_h);

        // 转置图像以匹配 Tensor 的存储格式（行列互换）
        const cv::Mat &split_image_t = split_image.t();

        // 将每个通道的数据复制到 Tensor 的对应切片中
        memcpy(input->slice(index).memptr(), split_image_t.data,
               sizeof(float) * split_image.total());
        index += 1;
    }

    // 定义 RGB 通道的均值
    float mean_r = 0.485f;
    float mean_g = 0.456f;
    float mean_b = 0.406f;

    // 定义 RGB 通道的标准差
    float var_r = 0.229f;
    float var_g = 0.224f;
    float var_b = 0.225f;

    // 确保输入的通道数为 3
    assert(input->channels() == 3);

    // 将图像数据归一化到 [0, 1] 范围
    input->data() = input->data() / 255.f;

    // 对每个通道进行均值和标准差归一化
    input->slice(0) = (input->slice(0) - mean_r) / var_r;
    input->slice(1) = (input->slice(1) - mean_g) / var_g;
    input->slice(2) = (input->slice(2) - mean_b) / var_b;

    // 返回预处理后的图像 Tensor
    return input;
}

TEST(test_network, resnet1) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model_files/simple_conv_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build({ "Input" }, { "Output" });
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

    auto outputs = graph.Forward({ input }, true);
    std::cout << outputs.size() << std::endl;
    outputs[0].at(0)->Show();
//    std::cout << "Hello World" << std::endl;
}

// 测试拓扑结构是否正常
TEST(test_network, resnet2) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model_files/resnet18.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build({ "Input" }, { "Output" });
    ASSERT_EQ(int(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor = std::make_shared<ftensor>(3, 2, 2);
    input_tensor->Ones();
    std::vector<sftensor> input{input_tensor};
    input.at(0)->Show();

    auto outputs = graph.Forward({ input }, true);
    outputs[0].at(0)->Show();
}

// 验证图像加载是否成功
TEST(test_network, resnet3) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model_files/resetnet18_batch1.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build({ "Input" }, { "Output" });
    ASSERT_EQ(int(graph.graph_state()), 0);

    const uint32_t batch_size = 1;
    std::vector<sftensor> inputs;
    const std::string &path("./model/car.jpg");

    for (uint32_t i = 0; i < batch_size; ++i) {
        cv::Mat image = cv::imread(path);
        // 图像预处理
        sftensor input = PreProcessImage(image);
        inputs.push_back(input);
    }
    auto outputs = graph.Forward({ inputs }, true);
    outputs[0].at(0)->Show();
    ASSERT_EQ(outputs.size(), batch_size);

    SoftmaxLayer softmax_layer(0);
    std::vector<sftensor> outputs_softmax(batch_size);
    softmax_layer.Forward(outputs[0], outputs_softmax);
    assert(outputs_softmax.size() == batch_size);

    for (int i = 0; i < outputs_softmax.size(); ++i) {
        const sftensor &output_tensor = outputs_softmax.at(i);
        assert(output_tensor->size() == 1 * 1000);
        // 找到类别概率最大的种类
        float max_prob = -1;
        int max_index = -1;
        for (int j = 0; j < output_tensor->size(); ++j) {
            float prob = output_tensor->index(j);
            if (max_prob <= prob) {
                max_prob = prob;
                max_index = j;
            }
        }
        printf("class with max prob is %f index %d\n", max_prob, max_index);
    }
}
