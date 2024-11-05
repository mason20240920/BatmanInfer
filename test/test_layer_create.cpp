//
// Created by Mason on 2024/10/30.
//
#include <layer/abstract/layer_factory.hpp>
#include <gtest/gtest.h>
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

    sftensor input_tensor = std::make_shared<ftensor>(3, 4, 4);
    input_tensor->Rand();
    input_tensor->data()-= 0.5f;

    LOG(INFO) << input_tensor->data();

    std::vector<sftensor> inputs(1);
    std::vector<sftensor> outputs(1);
    inputs.at(0) = input_tensor;
    auto start = std::chrono::high_resolution_clock::now();
    layer->Forward(inputs, outputs);
    auto end = std::chrono::high_resolution_clock ::now();
    // 计算持续时间
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Function execution time: " << duration.count() << " ms\n";
    for (const auto& output: outputs)
        output->Show();
}

TEST(test_registry, create_layer_softmax_forward) {
    // 1. 初始化一个运行时算子
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->type = "nn.Softmax";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegister::CreateLayer(op);
    ASSERT_NE(layer, nullptr);

    sftensor input_tensor = std::make_shared<ftensor>(1, 3, 4);
    input_tensor->Rand();
    input_tensor->data()-= 0.5f;

    LOG(INFO) << input_tensor->data();

    std::vector<sftensor> inputs(1);
    std::vector<sftensor> outputs(1);
    inputs.at(0) = input_tensor;
    layer->Forward(inputs, outputs);
    for (const auto& output: outputs)
        output->Show();
}
