//
// Created by Mason on 2024/10/30.
//
#include <layer/abstract/layer_factory.hpp>
#include <gtest/gtest.h>
#include <map>
#include <Halide.h>
using namespace BatmanInfer;

static LayerRegister::CreateRegistry *RegistryGlobal() {
    static auto *kRegistry = new LayerRegister::CreateRegistry();
    CHECK(kRegistry != nullptr) << "Global layer register init failed!";
    return kRegistry;
}

TEST(test_registry, registry1) {
    using namespace BatmanInfer;
    LayerRegister::CreateRegistry  *registry1 = RegistryGlobal();
    LayerRegister::CreateRegistry  *registry2 = RegistryGlobal();

    LayerRegister::CreateRegistry *registry3 = RegistryGlobal();
    LayerRegister::CreateRegistry *registry4 = RegistryGlobal();
    auto *a = new float{3};
    auto *b = new float{4};
    ASSERT_EQ(registry1, registry2);
}

ParseParameterAttrStatus MyTestCreator(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &layer) {

    layer = std::make_shared<Layer>("test_layer");
    return ParseParameterAttrStatus::bParameterAttrParseSuccess;
}

TEST(test_registry, registry2) {
    using namespace BatmanInfer;
    LayerRegister::CreateRegistry registry1 = LayerRegister::Registry();
    LayerRegister::CreateRegistry registry2 = LayerRegister::Registry();
    ASSERT_EQ(registry1, registry2);
    LayerRegister::RegisterCreator("test_type", MyTestCreator);
    LayerRegister::CreateRegistry registry3 = LayerRegister::Registry();
    ASSERT_EQ(registry3.size(), 3);
    ASSERT_NE(registry3.find("test_type"), registry3.end());
}

TEST(test_registry, create_layer) {
    LayerRegister::RegisterCreator("test_type_1", MyTestCreator);
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "test_type_1";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_util) {
    LayerRegistererWrapper kReluGetInstance("test_type_2", MyTestCreator);
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "test_type_2";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_relu_forward) {
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "nn.ReLU";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);

    sftensor input_tensor = std::make_shared<ftensor>(3, 4, 3);
    input_tensor->Rand();
}

TEST(test_registry, create_layer_softmax_forward) {
    // 1. 初始化一个运行时算子
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "Relu";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);

    sftensor input_tensor = std::make_shared<ftensor>( 1, 1, 8, 768);
    input_tensor->Rand();
    input_tensor->Show();
    std::map<std::string, sftensor> input_map{
            {"input", input_tensor}
    };

    sftensor output = std::make_shared<ftensor>(1, 1, 8, 768);
    std::map<std::string, sftensor> output_map {
            {"output", output},
    };

    layer->Forward(input_map, output_map);

    output_map.at("output")->Show();

}

TEST(test_registry, create_softmax_1) {
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "Softmax";
    std::string name = "axis";
    op->params = std::map<std::string, std::shared_ptr<RuntimeParameter>>{{name, std::make_shared<RuntimeParameterInt>(0)}};
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);

    sftensor input_tensor = std::make_shared<ftensor>(8, 8);
    input_tensor->Ones();
    input_tensor->Show();
    std::map<std::string, sftensor> input_map {
            {"input", input_tensor}
    };

    sftensor output = std::make_shared<ftensor>(8, 8);
    std::map<std::string, sftensor> output_map {
            {"output", output}
    };

    layer->Forward(input_map, output_map);

    output_map.at("output")->Show();

}

TEST(test_registry, test_halide_buffer) {
    using namespace Halide;
    const int N = 2, C = 3, H = 4, W = 4; // Example dimensions
    halide_dimension_t dimensions[4] = {
            {0, N, 1},        // Batch dimension
            {0, C, N},        // Channel dimension
            {0, H, N * C},    // Height dimension
            {0, W, N * C * H} // Width dimension
    };
    halide_buffer_t input = {0};
    input.dimensions = 4;
    input.dim = dimensions;
    input.type = halide_type_of<float>();
    float input_data[N * C * H * W];
    input.host = (uint8_t *)input_data;

    // Create an output buffer
    halide_buffer_t output = {0};
    output.dimensions = 4;
    output.dim = dimensions;
    output.type = halide_type_of<float>();
    float output_data[N * C * H * W];
    output.host = (uint8_t *)output_data;

    // Initialize input data
    for (int i = 0; i < N * C * H * W; i++) {
        input_data[i] = (i % 10) - 5; // Some test data
    }

    Buffer<float> in(input);

    Var n, c, h, w;

    // Step 3: Define the ReLU function
    Func relu;
    relu(n, c, h, w) = max(0.0f, in(n, c, h, w));

    Buffer<float> out(output);
    relu.realize(out);

    out.copy_to_host();

    // Print the output
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int index = n * C * H * W + c * H * W + h * W + w;
                    std::cout << "output[" << n << "][" << c << "][" << h << "][" << w << "] = "
                              << input_data[index] << "\n";
                }
            }
        }
    }
}
