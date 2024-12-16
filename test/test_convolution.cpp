//
// Created by Mason on 2024/11/12.
//

#include <gtest/gtest.h>
#include <layer/detail/convolution.hpp>
#include <Halide.h>
#include "runtime/runtime_ir.hpp"

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



TEST(ConvolutionLayerTest, TestInitIm2ColWeight) {
    // 调用 InitIm2ColWeight 函数ju
    using namespace BatmanInfer;
    const std::string& model_path = "../model_files/simple_conv_model.onnx";
    RuntimeGraph graph(model_path);
    ASSERT_EQ(int(graph.graph_state()), -2);
    const bool init_success = graph.Init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.graph_state()), -1);
    graph.Build({ "input" }, { "output" });
    ASSERT_EQ(int(graph.graph_state()), 0);

    std::shared_ptr<ftensor> input_tensor1 = std::make_shared<ftensor>(1, 2, 2, 2);
    input_tensor1->Ones();
    std::map<std::string, sftensor> input_from{{"input", input_tensor1}};

//    std::shared_ptr<ftensor> input_tensor2 = std::make_shared<ftensor>(1,1, 3, 3);
//    input_tensor2->Ones();
//    std::map<std::string, sftensor> output_from{{"output", input_tensor2}};

    auto outputs = graph.Forward(input_from, true);
}

TEST(test_registry, create_layer_conv_forward) {
    std::map<std::string, sftensor> inputs{{"input", std::make_shared<ftensor>(4, 4, 4)}};
    std::map<std::string, sftensor> outputs{{"output", nullptr}};


}

void print_buffer_properties_as_matrix(Halide::Buffer<float> buffer) {
    // Check if the buffer has at least 4 dimensions (batch, channel, height, width)
    if (buffer.dimensions() < 2) {
        std::cerr << "Error: Buffer must have at least 4 dimensions (batch, channel, height, width)!" << std::endl;
        return;
    }
    // Extract dimensions2
    int height = buffer.dim(1).extent();    // Height (rows)
    int width = buffer.dim(0).extent();     // Width (columns)

    // Print buffer contents as matrices for each batch and channel
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            std::cout << buffer(w, h) << "\t"; // Print each element with tab spacing
        }
        std::cout << std::endl; // Move to the next row
    }
}




std::vector<Halide::Buffer<float>> im2row_split(Halide::Buffer<float> input, const int group) {
    using namespace Halide;
    // Input dimensions: kernel_count (K), channel (C), rows (H), cols (W)
    int kernel_count = input.dim(3).extent();
    int channel = input.dim(2).extent();
    int rows = input.dim(1).extent();
    int cols = input.dim(0).extent();

    // Ensure the channel can be evenly divided by the group
    assert(kernel_count % group == 0 && "Channel count must be divisible by group");

    // 切分后的channel
    int split_kernel_count = kernel_count / group;

    // Output dimensions: kernel_count x (C * H * W)
    int flattened_size = channel * rows * cols;

    // To store the output buffers for each group
    std::vector<Halide::Buffer<float>> output_buffers;

    // Process each group separately
    for (int g = 0; g < group; ++g) {
        // Halide function for Im2Row transformation
        Func im2row;
        Var k, n;

        // Define the computation for the current group
        // Map the input indices to the appropriate group
        // Define the computation for the current group
        // Map the input indices to the appropriate group
        im2row(k, n) = input(
                n % cols,                            // W
                (n / cols) % rows,                   // H
                (n / (cols * rows)),                 // C
                g * split_kernel_count + k           // K
        );

        // Schedule: optimize for parallelism and vectorization if needed
        // im2row.parallel(k).vectorize(n, 16);

        // Realize the output buffer for the current group
        Buffer<float> output(split_kernel_count, flattened_size);
        im2row.realize(output);

        // Store the output buffer in the vector
        output_buffers.push_back(output);
    }

    return output_buffers;
}

TEST(test_conv, split_conv_kernel) {
    using namespace Halide;
// Example input: 2 kernels, 3 channels, 2x2 size
    Buffer<float> input(2, 2, 4, 2);

    int index = 0;

    // Fill the input with some values
    for (int k = 0; k < 2; ++k) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 4; ++h) {
                for (int w = 0; w < 2; ++w) {
                    input(k, c, h, w) = index;
                    index++;
                }
            }
        }
    }

    // Perform Im2Row split
    std::vector<Halide::Buffer<float>> result = im2row_split(input, 2);
    for (auto item: result)
        print_buffer_properties_as_matrix(item);
}